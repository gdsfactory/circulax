from circulax.compiler import compile_netlist


def test_compile_netlist_basic(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    # Basic expectations
    assert isinstance(port_map, dict)
    assert "V1,p2" in port_map and "C1,p1" in port_map

    # There should be a group for each non-ground component
    group_names = set(groups.keys())
    assert {"resistor", "capacitor", "inductor", "source_voltage"} <= group_names

    # Sys size should include node count + internal variables (V1.i_src and L1.i_L)
    # Compute num_nets from the port connection keys (excludes state entries)
    all_port_keys = set()
    for src, targets in net_dict["connections"].items():
        all_port_keys.add(src)
        for t in (targets if isinstance(targets, (list, tuple)) else [targets]):
            all_port_keys.add(t)
    num_nets = len({port_map[k] for k in all_port_keys if k in port_map})
    assert sys_size == num_nets + 2

    # Internal state indices are now exposed in port_map
    assert "V1,i_src" in port_map
    assert "L1,i_L" in port_map
