@echo off
setlocal

pushd asm

copy /y jump_i386_ms_pe_gas.asm jump_i386_ms_pe_gas.S
copy /y jump_x86_64_ms_pe_gas.asm jump_x86_64_ms_pe_gas.S
copy /y make_i386_ms_pe_gas.asm make_i386_ms_pe_gas.S
copy /y make_x86_64_ms_pe_gas.asm make_x86_64_ms_pe_gas.S
copy /y ontop_i386_ms_pe_gas.asm ontop_i386_ms_pe_gas.S
copy /y ontop_x86_64_ms_pe_gas.asm ontop_x86_64_ms_pe_gas.S

popd