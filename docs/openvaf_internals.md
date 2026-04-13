# OpenVAF Internals (OSDI 0.4)

> Source: https://github.com/robtaylor/OpenVAF/blob/feature/make-crates-usable-outside/internals.md

# Parameter storage

## Instance
- instance params (the ones flagged with type="instance")
- opvars

## Model
- instance params
- model params (the ones not flagged with type="instance")

## Given flags encoding (words, bits)
	words                         w0          w1          ...
	bits within a word            b31 ... b0  b31 ... b0  ...
	flags with parameter indices  f31 ... f0  f63 ... f32 ...


# Model data
	given flags: array of u32 values, count=floor((nmpar+31)/32)
	parameters: count=nmpar
		parameters that are not marked with type="instance"
		instance parameter fields from instance data (*)


# Instance data
	given_flags: array of u32 values, count=floor((nipar+31)/32)
	jacobian_ptr: array of num_jacobian_entries pointers to double, one for each resistive Jacobian entry
	jacobian_ptr_react: array of pointers to double, one for each reactive jacobian entry (with react_ptr_off!=UINT32_MAX)
	node_mapping: array of u32, count=nunknowns
	collapsed: array of i8, count=nnodepairs
	temperature: f64 ? what is its use?
	connected_ports: i32 - number of connected ports
	state_idx: array of i32, count=nstatesforlimiting

	params (*): count=nipar
		builtin instance params that are live (for now only $mfactor) type=???
		alias instance params (sys_fun_alias), 0 for now ??? type=???
		user instance parameters (marked with type="instance")
	cache
        intermediate results
        Jacobian contributions that are not constants or otherwise trivially determinable
	eval_outputs, order is not guaranteed, same values are stored only once
		opvars
		residuals
		bound_step


# Parameter fields
	integer .. i32
	real    .. f64
	string  .. pointer (u64)
	integer vector .. array of i32 values with given length (product across dimensions), length is static
	real vector    .. array of double values with given length (product across dimensions), length is static
	string vector  .. array of pointers with given length (product across dimensions), length is static


# OSDI 0.4 descriptor entries

## Given flags access

    uint32_t (*given_flag_model)(void *model, uint32_t id);

Returns true if given flag is set for model parameter with given OSDI id.
If id is not a valid model parameter, returns false.

    uint32_t (*given_flag_instance)(void *inst, uint32_t id);

Returns true if given flag is set for instance parameter with given OSDI id.
If id is not a valid instance parameter, returns false.


## Loading Jacobian entries into an array

These functions are used for extracting Jacobian contributions of a single instance.
They can be used for checking whether an instance is converged.

    uint32_t num_resistive_jacobian_entries;

Number of resistive Jacobian entries. Should be <= num_jacobian_entries.
This is the number of Jacobian entries with JACOBIAN_ENTRY_RESIST flag set.

    uint32_t num_reactive_jacobian_entries;

Number of reactive Jacobian entries. Should be <= num_jacobian_entries.
This is the number of Jacobian entries with JACOBIAN_ENTRY_REACT flag set.

    void write_jacobian_array_resist(void *inst, void* model, double* destination);

Writes resistive Jacobian contributions to an array of doubles of length num_resistive_jacobian_entries.
The entries appear in the same order as in jacobian_entries.
Entries with JACOBIAN_ENTRY_RESIST flag not set are left out.

    void write_jacobian_array_react(void *inst, void* model, double* destination);

Writes reactive Jacobian contributions to an array of doubles of length num_reactive_jacobian_entries.
The entries appear in the same order as in jacobian_entries.
Entries with JACOBIAN_ENTRY_REACT flag not set are left out.


## List of model inputs

Obtains the list of input quantities to a model (just like they appear in Verilog-A code).
This information is useful when checking whether an instance is converged.
Only inputs that cause nonlinear response are listed.

    uint32_t num_inputs;

Number of model inputs.

    OsdiNodePair* inputs;

Model inputs as node pairs. If a node in a node pair is UINT32_MAX,
it corresponds to global ground node.


## Loading Jacobian entries with an offset

These functions are used for adding Jacobian contributions of a single instance
to an address shifted by an offset from the given Jacobian entry pointers.
This is useful in analyses like HB where we need to evaluate the circuit at multiple
timepoints and store the results in vectors.

    void load_jacobian_with_offset_resist(void *inst, void* model, size_t offset);

Like load_jacobian_resist(), except that it adds an offset to the Jacobian entry pointer.
Offset of 1 means that the address in memory where the entry is added to is
the address pointed to by the Jacobian entry pointer increased by the size of one double (8).

    void load_jacobian_with_offset_react(void *inst, void* model, size_t offset);

Like load_jacobian_react(), except that it adds an offset to the Jacobian entry pointer.
Assumes alpha=1. Offset of 1 means that the address in memory where the entry is added to is
the address pointed to by the Jacobian entry pointer increased by the size of one double (8).


# OSDI 0.4 symbols in the generated dynamic library.

    OSDI_DESCRIPTOR_SIZE

Size of the OSDI descriptor in bytes. Can be used by simulators supporting only
OSDI 0.3 for traversing the array of descriptors. The first part of the descriptor
is compatible with OSDI 0.3.
