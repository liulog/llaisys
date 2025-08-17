target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")

    -- 非 Windows：仅启用 -O3（GCC/Clang 下会自动开启向量化）
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", "-O3", "-fopt-info-vec-optimized")
    end
 
    -- Windows：
    -- MSVC 用 /O2；MinGW/Clang on Windows 仍用 -O3
    if is_plat("windows") then
        add_cxflags("/O2", {tools = "cl"})       -- MSVC
        add_cxflags("-O3", {tools = "gcc"})      -- MinGW
        add_cxflags("-O3", {tools = "clang"})    -- LLVM clang-cl/clang++
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()

