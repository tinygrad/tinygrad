from tinygrad import Tensor, Device

a = Tensor.empty(1, device="NV").add(1)
ei = a.schedule()[0].lower()

print(len(ei.prg.p.lib))
Device[a.device].compiler.disassemble(ei.prg.p.lib)
ei.run()
