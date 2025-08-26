-- CUDA Device target
target("llaisys-device-nv")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    ---------------------------------------------------
    -- Important! Relocatable Device Code
    set_values("cuda.rdc", false)
    ---------------------------------------------------

    add_cugencodes("native")

    if not is_plat("windows") then
        add_cuflags("-O3", "-Xcompiler=-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()


target("llaisys-ops-nv")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")

    ---------------------------------------------------
    -- Important! Relocatable Device Code
    set_values("cuda.rdc", false)
    ---------------------------------------------------

    add_cugencodes("native")

    if not is_plat("windows") then
        add_cuflags("-O3", "-Xcompiler=-fPIC")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
