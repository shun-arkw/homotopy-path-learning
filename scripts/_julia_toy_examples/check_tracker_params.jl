using HomotopyContinuation

p_def  = HomotopyContinuation.DEFAULT_TRACKER_PARAMETERS
p_fast = HomotopyContinuation.FAST_TRACKER_PARAMETERS
p_con  = HomotopyContinuation.CONSERVATIVE_TRACKER_PARAMETERS

fields = fieldnames(typeof(p_fast))

for f in fields
    vdef  = getfield(p_def, f)
    vfast = getfield(p_fast, f)
    vcon  = getfield(p_con, f)
    println(rpad(string(f), 28), "  default=", vdef, "  fast=", vfast, "  cons=", vcon)
end
