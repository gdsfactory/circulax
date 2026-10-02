"""circulax netlists.

Supports both kfnetlist.Netlist (primary) and SAX-format dicts (backward compat).
SAX dicts are converted to kfnetlist.Netlist via :func:`sax_to_kfnetlist` before
node-index assignment.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from typing import Annotated, Any, NotRequired, TypeAlias

import kfnetlist as kfnl
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sax
import sax.saxtypes.netlist as sax_netlist
from natsort import natsorted
from sax.saxtypes import Instances, Placements, Ports
from sax.saxtypes.core import bval
from sax.saxtypes.settings import Settings
from sax.saxtypes.singlemode import InstancePort
from typing_extensions import TypedDict

Connections: TypeAlias = dict[InstancePort, InstancePort | tuple[InstancePort, ...]]

circulaxNetlist = Annotated[
    TypedDict(
        "Netlist",
        {
            "instances": Instances,
            "connections": NotRequired[Connections],
            "ports": Ports,
            "placements": NotRequired[Placements],
            "settings": NotRequired[Settings],
        },
    ),
    bval(sax_netlist.val_netlist),
]
"""Legacy SAX-format netlist type. Prefer ``kfnetlist.Netlist`` for new code."""

SaxNetlist = circulaxNetlist
Netlist = kfnl.Netlist


# ---------------------------------------------------------------------------
# kfnetlist-native node mapping
# ---------------------------------------------------------------------------


def build_net_map_kfnetlist(nl: kfnl.Netlist) -> tuple[dict[str, int], int]:
    """Map every port in a kfnetlist.Netlist to a node index.

    Each ``Net`` in the netlist becomes one electrical node.  Nets
    containing a ground instance port, or a synthetic net label containing
    ``"GND"``, are assigned node index 0.

    Returns:
        port_to_idx: ``{"Instance,port": node_id, ...}``
        num_nets: next free node index (== number of non-ground nets + 1).

    """
    port_to_idx: dict[str, int] = {}
    current_idx = 1

    for net in nl.nets:
        is_ground = any(
            (
                isinstance(m, kfnl.PortRef)
                and (m.instance == "GND" or (m.instance in nl.instances and nl.instances[m.instance].component == "ground"))
            )
            or (isinstance(m, kfnl.NetlistPort) and "GND" in m.name)
            for m in net
        )
        net_id = 0 if is_ground else current_idx

        for member in net:
            if isinstance(member, kfnl.PortRef):
                port_to_idx[f"{member.instance},{member.port}"] = net_id
            elif isinstance(member, kfnl.NetlistPort):
                port_to_idx[member.name] = net_id

        if not is_ground:
            current_idx += 1

    return port_to_idx, current_idx


def prune_unreachable_instances(netlist: kfnl.Netlist | dict) -> kfnl.Netlist | dict:
    """Drop instances that cannot influence the circuit's exposed ports.

    An instance only matters to the simulation if some signal path connects
    it (directly or transitively, through shared nets) to one of its
    circuit's top-level ports. Structural/decorative cells (die frames,
    logos, marker rectangles) are typically wired to nothing — ``"ports": []``
    and no nets — so they take part in no signal path. Left in place, such a
    cell can abort compilation (or, upstream, an unmodeled-component walk
    that treats "no model" as "must be a composite to expand") with a
    spurious "Model not found" error for a component nobody asked to
    simulate. See gdsfactory/circulax#60.

    Accepts a ``kfnl.Netlist``, a flat SAX-format dict, or a
    ``RecursiveNetlist`` (dict of SAX dicts, i.e. subcircuit bodies keyed by
    name). For a flat netlist (kfnetlist or dict), the whole prune is skipped
    if it declares no top-level ports at all — with none, every instance
    would look unreachable, and a netlist driven purely by an internal source
    (e.g. a lone oscillator with no ``ports`` declared) would be emptied out.

    For a RecursiveNetlist, that same guard applies only to the top circuit
    (the first key); once it passes, every sub-netlist — top included — is
    pruned against its own declared ports with no further guard, so a
    sub-netlist with no ports of its own is emptied out entirely. This
    matches ``sax.netlists.remove_unused_instances`` and removes nested
    inert cells consistently.

    Reimplemented natively here (rather than delegating to
    ``sax.netlists.remove_unused_instances``) because circulax's own
    ``connections`` values may be a tuple/list of targets (one shared net for
    several refs), which sax's dict-oriented helper does not support.

    Returns:
        A new netlist of the same type as *netlist*, with unreachable
        instances (and the connections/nets that only touched them) removed.
        The input is not mutated.

    """
    if isinstance(netlist, kfnl.Netlist):
        return _prune_unreachable_kfnetlist(netlist)

    if "instances" in netlist:
        if not netlist.get("ports"):
            return netlist
        return _prune_unreachable_sax_dict(netlist)

    top_name = next(iter(netlist), None)
    if top_name is None:
        return netlist
    top = netlist.get(top_name) or {}
    if not top.get("ports"):
        return netlist

    return {name: _prune_unreachable_sax_dict(net) for name, net in netlist.items()}


def _sax_ref_key(ref: str) -> str:
    """Union-find key for a SAX port reference string (e.g. ``"R1,p1"`` -> ``"R1"``).

    The part before the comma identifies the electrical bridge: either a
    declared instance, or a hierarchy-stub label (e.g. ``"vdd,p1"``) that
    isn't a real instance at all. Either way, all ports sharing that prefix
    are electrically the same component for reachability purposes.
    """
    return ref.split(",", 1)[0] if "," in ref else ref


def _prune_unreachable_sax_dict(net: dict) -> dict:
    """Prune one flat SAX dict against its own declared ports.

    No "skip if portless" guard: a netlist with no ``ports`` has every
    instance marked unreachable and is emptied out, matching
    ``sax.netlists.remove_unused_instances``. Callers that want to protect a
    portless *top* circuit from this must guard before calling in (see
    :func:`prune_unreachable_instances`).
    """
    uf = _UnionFind()
    port_roots = {_sax_ref_key(target) for target in net.get("ports", {}).values()}
    for root in port_roots:
        uf.find(root)

    for src, targets in net.get("connections", {}).items():
        if not isinstance(targets, (list, tuple)):
            targets = (targets,)
        src_key = _sax_ref_key(src)
        for tgt in targets:
            uf.union(src_key, _sax_ref_key(tgt))

    for entry in net.get("nets", []):
        uf.union(_sax_ref_key(entry["p1"]), _sax_ref_key(entry["p2"]))

    reachable = {uf.find(root) for root in port_roots}

    def _is_live(key: str) -> bool:
        return uf.find(key) in reachable

    unreachable = {name for name in net.get("instances", {}) if not _is_live(name)}

    if not unreachable:
        return net

    pruned = dict(net)
    pruned["instances"] = {k: v for k, v in net["instances"].items() if k not in unreachable}
    if "connections" in net:
        # Filtered by the *root* of the source key (not "is this an instance
        # name in `unreachable`"), so a connection off a hierarchy-stub hub
        # label (e.g. "n1,p1": ("X1,p1", "X2,p1")) is dropped too when that
        # hub's whole clique is unreachable — not just the instances in it.
        pruned["connections"] = {src: targets for src, targets in net["connections"].items() if _is_live(_sax_ref_key(src))}
    if "nets" in net:
        pruned["nets"] = [entry for entry in net["nets"] if _is_live(_sax_ref_key(entry["p1"]))]
    return pruned


class _UnionFind:
    """Minimal union-find over string keys, local to reachability walks."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _kfnetlist_member_key(member: kfnl.PortRef | kfnl.NetlistPort) -> str:
    """Union-find key for a net member: the owning instance, or the port name."""
    if isinstance(member, kfnl.PortRef):
        return member.instance
    return f"\0port\0{member.name}"


def _drop_kfnetlist_instances(nl: kfnl.Netlist, names: set[str]) -> kfnl.Netlist:
    """Remove *names* from a copy of *nl*, along with any now-empty nets."""
    pruned = kfnl.Netlist.from_dict(nl.to_dict())
    pruned.remove_instances(sorted(names))
    cleaned = pruned.to_dict()
    cleaned["nets"] = [n for n in cleaned["nets"] if n]
    return kfnl.Netlist.from_dict(cleaned)


def _prune_unreachable_kfnetlist(nl: kfnl.Netlist) -> kfnl.Netlist:
    """kfnetlist-native half of :func:`prune_unreachable_instances`."""
    # sax_to_kfnetlist synthesizes a top-level port (named "inst,port") for
    # any connection that targets an unknown instance — a hierarchy-stub
    # label, not something the original netlist actually declared as a port.
    # If every port on `nl` looks synthetic, treat it the same as "no ports"
    # rather than pruning against those stub labels.
    if not nl.ports or all("," in p.name for p in nl.ports):
        return nl

    uf = _UnionFind()
    port_roots = {_kfnetlist_member_key(p) for p in nl.ports}
    for root in port_roots:
        uf.find(root)

    for net in nl.nets:
        keys = [_kfnetlist_member_key(m) for m in net]
        for a, b in itertools.pairwise(keys):
            uf.union(a, b)

    reachable = {uf.find(root) for root in port_roots}
    unreachable = {name for name in nl.instances if uf.find(name) not in reachable}

    if not unreachable:
        return nl
    return _drop_kfnetlist_instances(nl, unreachable)


# ---------------------------------------------------------------------------
# SAX → kfnetlist converter
# ---------------------------------------------------------------------------


def _is_json_safe(v: Any) -> bool:
    """Return True if *v* can survive a round-trip through serde_json."""
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def sax_to_kfnetlist(
    sax_dict: dict,
) -> tuple[kfnl.Netlist, dict[str, dict[str, Any]]]:
    """Convert a SAX-format netlist dict to a ``kfnetlist.Netlist``.

    Pairwise SAX ``connections`` (and GDSFactory-style ``nets`` lists)
    are grouped into equivalence classes via union-find, then emitted as
    one ``kfnetlist.Net`` per class.

    Returns:
        A 2-tuple ``(netlist, settings_override)`` where
        *settings_override* maps instance names to their full settings
        dict (including non-JSON-safe values such as complex numbers).
        When all settings are JSON-safe the override dict is empty.

    """
    nl = kfnl.Netlist()
    settings_override: dict[str, dict[str, Any]] = {}

    # --- instances ---
    for name, data in sax_dict.get("instances", {}).items():
        raw_settings = data.get("settings") or {}
        safe_settings = {k: v for k, v in raw_settings.items() if _is_json_safe(v)}
        nl.create_inst(
            name=name,
            kcl="",
            component=data["component"],
            settings=safe_settings or None,
        )
        if len(safe_settings) != len(raw_settings):
            settings_override[name] = raw_settings

    # --- ports ---
    declared_ports: set[str] = set()
    top_port_targets: list[tuple[str, str]] = []
    for port_name, target in sax_dict.get("ports", {}).items():
        nl.create_port(port_name)
        declared_ports.add(port_name)
        top_port_targets.append((port_name, target))

    # --- connectivity: union-find over SAX connections + GDSFactory nets ---
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    known_instances = set(sax_dict.get("instances", {}))

    def _net_member_ref(port_str: str) -> kfnl.PortRef | kfnl.NetlistPort:
        if "," not in port_str:
            msg = f"Expected SAX port reference 'instance,port', got {port_str!r}"
            raise ValueError(msg)
        inst, _port = port_str.split(",", 1)
        if inst not in known_instances:
            if port_str not in declared_ports:
                nl.create_port(port_str)
                declared_ports.add(port_str)
            return kfnl.NetlistPort(port_str)
        return kfnl.PortRef(instance=inst, port=_port)

    for src, targets in sax_dict.get("connections", {}).items():
        if isinstance(targets, str):
            targets = [targets]
        for tgt in targets:
            _find(src)
            _find(tgt)
            _union(src, tgt)

    for net_entry in sax_dict.get("nets", []):
        p1, p2 = net_entry["p1"], net_entry["p2"]
        _find(p1)
        _find(p2)
        _union(p1, p2)

    for _port_name, target in top_port_targets:
        _find(target)

    # Group ports by their root → one kfnetlist.Net per group
    groups: dict[str, list[str]] = defaultdict(list)
    for port in parent:
        groups[_find(port)].append(port)

    top_ports_by_root: dict[str, list[str]] = defaultdict(list)
    for port_name, target in top_port_targets:
        top_ports_by_root[_find(target)].append(port_name)

    for root, members in groups.items():
        refs = [kfnl.NetlistPort(port_name) for port_name in top_ports_by_root.get(root, [])]
        for port_str in sorted(members):
            refs.append(_net_member_ref(port_str))
        nl.create_net(*refs)

    nl.sort()
    return nl, settings_override


def kfnetlist_to_sax(nl: kfnl.Netlist) -> dict:
    """Adapt canonical topology for legacy SAX subcircuit composition only.

    kfnetlist.to_dict() uses kfnetlist's serialization schema, not SAX's.
    Explicitly translate net members and external-port aliases at this boundary.
    """
    result: dict = {
        "instances": {
            name: {"component": instance.component, "settings": instance.settings} for name, instance in nl.instances.items()
        },
        "nets": [],
        "ports": {},
    }
    for net in nl.nets:
        refs = [f"{m.instance},{m.port}" for m in net if isinstance(m, kfnl.PortRef)]
        ports = [m.name for m in net if isinstance(m, kfnl.NetlistPort)]
        if not refs:
            if ports:
                msg = f"Cannot embed a net containing only external ports: {ports}"
                raise ValueError(msg)
            continue
        result["nets"].extend({"p1": refs[0], "p2": ref} for ref in (refs[1:] or refs[:1]))
        for port in ports:
            result["ports"][port] = refs[0]
    return result


# ---------------------------------------------------------------------------
# Recursive netlist flattening
# ---------------------------------------------------------------------------


def _is_recursive_netlist(net_dict: dict) -> bool:
    """Return True if *net_dict* looks like a RecursiveNetlist (dict-of-Netlists)."""
    if "instances" in net_dict:
        return False
    return any(isinstance(v, kfnl.Netlist) or (isinstance(v, dict) and "instances" in v) for v in net_dict.values())


def _normalize_simulator_ground(flat: kfnl.Netlist) -> kfnl.Netlist:
    # All simulator ground terminals refer to the same node, including grounds
    # introduced by nested cells. Keep the established GND instance name.
    data = flat.to_dict()
    grounds = {name for name, instance in data["instances"].items() if instance["component"] == "ground"}
    if grounds:
        if "GND" in data["instances"] and "GND" not in grounds:
            msg = "Instance name 'GND' is reserved for the ground component"
            raise ValueError(msg)
        ground = data["instances"][sorted(grounds)[0]]
        data["instances"] = {name: inst for name, inst in data["instances"].items() if name not in grounds}
        data["instances"]["GND"] = ground
        for net in data["nets"]:
            for member in net:
                if member.get("instance") in grounds:
                    member["instance"] = "GND"
        flat = kfnl.Netlist.from_dict(data)
    return flat


def _flatten_circuit_netlists(recnet: dict[str, dict | kfnl.Netlist], sep: str = "~") -> kfnl.Netlist | dict:
    """Flatten canonical topology with kfnetlist; adapt legacy settings if needed."""
    netlists: dict[str, kfnl.Netlist] = {}
    overrides: dict[str, dict[str, Any]] = {}
    # kfnetlist settings are JSON values. Retain legacy arrays/complex values
    # outside topology, using instance info to follow their origin through flatten.
    for cell, source in recnet.items():
        if isinstance(source, kfnl.Netlist):
            netlists[cell] = source
            continue
        native, settings = sax_to_kfnetlist(source)
        if settings:
            data = native.to_dict()
            for name, values in settings.items():
                origin = str(len(overrides))
                overrides[origin] = values
                data["instances"][name].setdefault("info", {})["_circulax_settings_origin"] = origin
            native = kfnl.Netlist.from_dict(data)
        netlists[cell] = native
    maps = {
        cell: {name: instance.component for name, instance in net.instances.items() if instance.component in netlists}
        for cell, net in netlists.items()
    }
    top = next(iter(netlists))
    flat = netlists[top].flatten(
        netlists,
        instance_cell_map=maps[top],
        sub_instance_cell_maps=maps,
        recursive=True,
        allow_unconnected_ports=True,
        separator=sep,
    )
    flat = _normalize_simulator_ground(flat)
    flat.sort()
    if overrides:
        from copy import deepcopy

        legacy = kfnetlist_to_sax(flat)
        for name, instance in flat.instances.items():
            origin = instance.info.get("_circulax_settings_origin")
            if origin in overrides:
                legacy["instances"][name]["settings"] = deepcopy(overrides[origin])
        return legacy
    return flat


def flatten_recursive_netlist(recnet: dict[str, dict | kfnl.Netlist], sep: str = "~") -> dict:
    """Flatten a legacy cell mapping using kfnetlist's recursive flatten API.

    The first entry is the top cell. Component names select child cells and
    ``sep`` prefixes flattened instances. The return value stays a SAX-format
    dictionary for compatibility; compilation uses the canonical result directly.
    """
    flat = _flatten_circuit_netlists(recnet, sep)
    return kfnetlist_to_sax(flat) if isinstance(flat, kfnl.Netlist) else flat


# ---------------------------------------------------------------------------
# Legacy SAX build_net_map (kept for backward compat / draw_circuit_graph)
# ---------------------------------------------------------------------------


def build_net_map(netlist: dict) -> tuple[dict[str, int], int]:
    """Maps every port (e.g. 'R1,p1') to a generic Node Index (integer).

    Returns:
        port_to_idx: dict mapping 'Instance,Pin' -> int index
        num_nets: Total number of unique electrical nodes (excluding Ground).

    """
    g = nx.Graph()

    for src, targets in netlist.get("connections", {}).items():
        if isinstance(targets, str):
            targets = [targets]
        for tgt in targets:
            g.add_edge(src, tgt)

    for net in netlist.get("nets", []):
        g.add_edge(net["p1"], net["p2"])

    components = list(nx.connected_components(g))
    components.sort(key=lambda x: natsorted(list(x))[0])

    port_to_idx = {}
    current_idx = 1

    for comp in components:
        is_ground = any("GND" in node for node in comp)
        net_id = 0 if is_ground else current_idx

        for node in comp:
            port_to_idx[node] = net_id

        if not is_ground:
            current_idx += 1

    return port_to_idx, current_idx


def draw_circuit_graph(  # noqa: C901, PLR0912, PLR0915
    netlist: dict[str, dict] | kfnl.Netlist,
    layout_attempts: int = 10,
    *,
    show: bool = True,
) -> mpl.figure.Figure:
    """Visualize a circuit netlist as a connectivity graph.

    Accepts either a SAX-format dict or a ``kfnetlist.Netlist``.

    Args:
        netlist: Circuit description (SAX dict or kfnetlist.Netlist).
        show: If ``True``, call ``plt.show()`` before returning.
        layout_attempts: Number of spring-layout seeds to try.

    Returns:
        The :class:`matplotlib.figure.Figure` containing the rendered graph.

    """
    if isinstance(netlist, kfnl.Netlist):
        port_map, _ = build_net_map_kfnetlist(netlist)
        instance_names = list(netlist.instances)
    else:
        port_map, _ = build_net_map(netlist)
        instance_names = list(netlist.get("instances", {}))

    G = nx.Graph()

    for name in instance_names:
        if name == "GND":
            G.add_node(name, color="black", size=1500, label=name)
        else:
            G.add_node(name, color="red", size=2000, label=name)

    net_groups = {}

    for port_str, net_idx in port_map.items():
        if "," not in port_str:
            continue

        inst_name, pin_name = port_str.split(",", 1)

        G.add_node(port_str, color="skyblue", size=300, label=pin_name, parent=inst_name)

        if inst_name in G.nodes:
            G.add_edge(inst_name, port_str, weight=10, type="internal")

        if net_idx not in net_groups:
            net_groups[net_idx] = []
        net_groups[net_idx].append(port_str)

    edge_labels = {}

    for net_idx, ports in net_groups.items():
        if len(ports) > 1:
            for i in range(len(ports) - 1):
                u, v = ports[i], ports[i + 1]
                G.add_edge(u, v, weight=1, type="external")
                edge_labels[(u, v)] = str(net_idx)

    def make_initial_pos(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        pos = {}
        instance_nodes = [n for n, d in G.nodes(data=True) if d.get("color") in ["red", "black"]]

        # Place instances on a rough grid / random spread
        for name in instance_nodes:
            pos[name] = rng.uniform(-1, 1, size=2)

        # Place each port with a small random offset from its parent
        for n, d in G.nodes(data=True):
            if d.get("color") == "skyblue":
                parent = d.get("parent")
                if parent and parent in pos:
                    pos[n] = pos[parent] + rng.uniform(-0.1, 0.1, size=2)
                else:
                    pos[n] = rng.uniform(-1, 1, size=2)
        return pos

    def count_crossings(pos: dict[str, np.ndarray]) -> int:

        def segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:

            def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
                return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)

            return bool(((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))

        edges = list(G.edges())
        crossings = 0
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                u1, v1 = edges[i]
                u2, v2 = edges[j]
                # Skip pairs that share a node (adjacent edges always "meet", not cross)
                if u1 in (u2, v2) or v1 in (u2, v2):
                    continue
                p1, p2 = pos[u1], pos[v1]
                p3, p4 = pos[u2], pos[v2]
                if segments_intersect(p1, p2, p3, p4):
                    crossings += 1
        return crossings

    best_pos = None
    best_crossings = float("inf")

    for attempt in range(layout_attempts):
        seed = attempt  # deterministic across runs with same layout_attempts
        init_pos = make_initial_pos(seed)
        candidate_pos = nx.spring_layout(
            G,
            pos=init_pos,  # warm start near the desired structure
            fixed=None,  # let everything move, but start well
            k=0.5,
            iterations=80,  # more iterations since we have a good warm start
            weight="weight",
            seed=seed,
        )
        crossings = count_crossings(candidate_pos)
        if crossings < best_crossings:
            best_crossings = crossings
            best_pos = candidate_pos

    pos = best_pos

    fig = plt.figure(figsize=(10, 8))

    instance_nodes = [n for n, d in G.nodes(data=True) if d.get("color") in ["red", "black"]]
    port_nodes = [n for n, d in G.nodes(data=True) if d.get("color") == "skyblue"]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=instance_nodes,
        node_color=[G.nodes[n]["color"] for n in instance_nodes],
        node_size=[G.nodes[n]["size"] for n in instance_nodes],
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: n for n in instance_nodes},
        font_color="white",
        font_weight="bold",
    )

    nx.draw_networkx_nodes(G, pos, nodelist=port_nodes, node_color="skyblue", node_size=300)

    port_labels = {n: G.nodes[n]["label"] for n in port_nodes}
    nx.draw_networkx_labels(G, pos, labels=port_labels, font_size=8, font_color="black")

    internal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "internal"]
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, width=2.0, alpha=0.5)

    external_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "external"]
    nx.draw_networkx_edges(G, pos, edgelist=external_edges, width=1.5, style="dashed", edge_color="gray")

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="blue")

    ax = plt.gca()
    ax.set_title("Circuit Connectivity Graph")
    ax.axis("off")
    fig.tight_layout()

    if show:
        plt.show()

    return fig


# SAX type aliases kept for backward compatibility
Port = sax.Port
Ports = sax.Ports
Net = sax.Net
Nets = sax.Nets
Placements = sax.Placements
Instances = sax.Instances

# Primary netlist type is now kfnetlist.Netlist
netlist = kfnl.Netlist
