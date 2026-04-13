set(device_sources
    "opamp.va"           "opamp.osdi"            ""
    "capacitor.va"       "capacitor.osdi"        ""
    "inductor.va"        "inductor.osdi"         ""
    "resistor.va"        "resistor.osdi"         ""
    "diode.va"           "diode.osdi"            ""
    "bsim3v3.va"         "bsim3v3.osdi"          ""
    "bsim4v8.va"         "bsim4v8.osdi"          ""
    "bsimbulk106.va"     "bsimbulk106.osdi"      ""
    "vbic/vbic_1p3.va"   "vbic_1p3.osdi"         "-D__NGSPICE__ -DthreeTerminal"
    "vbic/vbic_1p3.va"   "vbic_1p3_4t.osdi"      "-D__NGSPICE__ -DfourTerminal"
    "vbic/vbic_1p3.va"   "vbic_1p3_5t.osdi"      "-D__NGSPICE__ -DfiveTerminal"
    "psp103v4/psp103.va" "psp103v4.osdi"         ""
    "spice/resistor.va"  "spice/resistor.osdi"   ""
    "spice/capacitor.va" "spice/capacitor.osdi"  ""
    "spice/inductor.va"  "spice/inductor.osdi"   ""
    "spice/diode.va"     "spice/diode.osdi"      ""
    "spice/jfet1.va"     "spice/jfet1.osdi"      ""
    "spice/jfet2.va"     "spice/jfet2.osdi"      ""
    "spice/mes1.va"      "spice/mes1.osdi"       ""
    "spice/bjt.va"       "spice/bjt.osdi"        ""
    "spice/mos1.va"      "spice/mos1.osdi"       ""
    "spice/mos2.va"      "spice/mos2.osdi"       ""
    "spice/mos3.va"      "spice/mos3.osdi"       ""
    "spice/mos6.va"      "spice/mos6.osdi"       ""
    "spice/mos9.va"      "spice/mos9.osdi"       ""
    "spice/vdmos.va"     "spice/vdmos.osdi"       ""
    "spice/bsim3v3.va"   "spice/bsim3v3.osdi"    ""
    "spice/bsim4v8.va"   "spice/bsim4v8.osdi"    ""
    "spice/sn/diode.va"     "spice/sn/diode.osdi"  ""
    "spice/sn/jfet1.va"     "spice/sn/jfet1.osdi"  ""
    "spice/sn/jfet2.va"     "spice/sn/jfet2.osdi"  ""
    "spice/sn/mes1.va"      "spice/sn/mes1.osdi"   ""
    "spice/sn/bjt.va"       "spice/sn/bjt.osdi"    ""
    "spice/sn/mos1.va"      "spice/sn/mos1.osdi"   ""
    "spice/sn/mos2.va"      "spice/sn/mos2.osdi"   ""
    "spice/sn/mos3.va"      "spice/sn/mos3.osdi"   ""
    "spice/sn/mos6.va"      "spice/sn/mos6.osdi"   ""
    "spice/sn/mos9.va"      "spice/sn/mos9.osdi"   ""
    "spice/full/capacitor.va" "spice/full/capacitor.osdi"  ""
    "spice/full/inductor.va"  "spice/full/inductor.osdi"   ""
    "spice/full/diode.va"     "spice/full/diode.osdi"      ""
    "spice/full/jfet1.va"     "spice/full/jfet1.osdi"      ""
    "spice/full/jfet2.va"     "spice/full/jfet2.osdi"      ""
    "spice/full/mes1.va"      "spice/full/mes1.osdi"       ""
    "spice/full/bjt.va"       "spice/full/bjt.osdi"        ""
    "spice/full/mos1.va"      "spice/full/mos1.osdi"       ""
    "spice/full/mos2.va"      "spice/full/mos2.osdi"       ""
    "spice/full/mos3.va"      "spice/full/mos3.osdi"       ""
    "spice/full/mos6.va"      "spice/full/mos6.osdi"       ""
    "spice/full/mos9.va"      "spice/full/mos9.osdi"       ""
    "spice/full/vdmos.va"     "spice/full/vdmos.osdi"      ""
    "spice/full/bsim3v3.va"   "spice/full/bsim3v3.osdi"    ""
)

# Extra options for OpenVAF
message(STATUS "Extra OpenVAF options: ${OPENVAF_OPTIONS}")
string(REPLACE " " ";" extra_options_list "${OPENVAF_OPTIONS}")

# Prepare directory for staging the osdi library
set(STAGED_OSDI_DIRECTORY "${STAGED_LIBRARY_DIRECTORY}/mod")
file(MAKE_DIRECTORY "${STAGED_OSDI_DIRECTORY}")

# Set upt building and staging
set(osdi_built_files "")
set(osdi_built_files_rel "")
set(osdi_staged_files "")
set(osdi_solurce_files "")
set(osdi_subdirs "")
list(LENGTH device_sources list_len)
set(i 0)
while(i LESS ${list_len})
    list(GET device_sources ${i} source)
    math(EXPR i "${i} + 1")
    list(GET device_sources ${i} output)
    math(EXPR i "${i} + 1")
    list(GET device_sources ${i} options)
    string(REPLACE " " ";" options_list "${options}")
    set(inputfile "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    # Add to source files
    list(APPEND osdi_source_files "${source}")
    # Destination is <dir_path>/<basename>.osdi
    # Temporary build directory is build-<dir_path> (depth may be >1)
    # Get <basename>
    get_filename_component(tmpnamebase "${output}" NAME_WLE)
    # Get <dir_path>
    get_filename_component(tmpdirectory "${output}" DIRECTORY)
    if ("${tmpdirectory}" STREQUAL "")
        # No <dir_path>
        set(tmpdirectory ".")
        set(builddir "${CMAKE_CURRENT_BINARY_DIR}/build-${tmpnamebase}")
    else()
        # Have <dir_path>
        set(builddir "${CMAKE_CURRENT_BINARY_DIR}/${tmpdirectory}/build-${tmpnamebase}")
    endif()
    if (NOT "${tmpdirectory}" EQUAL "." AND NOT "${tmpdirectory}" IN_LIST osdi_subdirs)
        list(APPEND osdi_subdirs "${tmpdirectory}")
        # Create directory where the built device will be stored
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${tmpdirectory}")
        # Create directory where the built device will be staged
        file(MAKE_DIRECTORY "${STAGED_OSDI_DIRECTORY}/${tmpdirectory}")
    endif()
    # message(STATUS ${builddir})
    # Full path to built file
    set(built_file "${CMAKE_CURRENT_BINARY_DIR}/${output}")
    # Full path to staged file
    set(staged_file "${STAGED_OSDI_DIRECTORY}/${output}")
    # Just the output file name
    set(outputfile "${tmpnamebase}.osdi")
    # Create build directory
    file(MAKE_DIRECTORY "${builddir}")
    # Build command - invoke openvaf, move to destination
    # message(STATUS "${OPENVAF_COMPILER} ${options} --allow variant_const_simparam -I${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${source} -o ${outputfile}")
    add_custom_command(
        OUTPUT "${built_file}"
        DEPENDS "${inputfile}"
        DEPENDS "${OPENVAF_COMPILER}"
        MAIN_DEPENDENCY "${inputfile}"
        COMMAND "${OPENVAF_COMPILER}" ARGS "${extra_options_list}" "${options_list}" "--allow" "variant_const_simparam" "-I${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/${source}" -o "${outputfile}"
        COMMAND ${CMAKE_COMMAND} ARGS -E copy "${outputfile}" "${built_file}"
        COMMAND ${CMAKE_COMMAND} ARGS -E rm -f "${outputfile}"
        COMMAND_EXPAND_LISTS
        WORKING_DIRECTORY "${builddir}"
        COMMENT "Building osdi model from ${inputfile}"
    )
    list(APPEND osdi_built_files "${built_file}")
    list(APPEND osdi_built_files_rel "${output}")
    add_custom_command(
        OUTPUT "${staged_file}"
        DEPENDS "${built_file}"
        COMMAND ${CMAKE_COMMAND} ARGS -E copy "${built_file}" "${staged_file}"
        COMMENT "Staging OSDI file ${outputfile}"
    )
    list(APPEND osdi_staged_files "${staged_file}")
    math(EXPR i "${i} + 1")
endwhile()

add_custom_target(
    build_and_stage_devices ALL
    SOURCES "${osdi_source_files}"
    DEPENDS "${osdi_staged_files}" "${osdi_built_files}"
)

SET(OSDI_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE STRING "Directory where OSDI files are stored" FORCE)
SET(OSDI_FILES "${osdi_built_files_rel}" CACHE STRING "List of OSDI files relative to OSDI_DIR" FORCE)
