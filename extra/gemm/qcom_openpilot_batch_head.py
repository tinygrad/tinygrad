#!/usr/bin/env python3
"""Batch adjacent independent openpilot head kernels into one QCOM launch."""
import argparse, pickle, re
from dataclasses import replace

from tinygrad import Device
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name

MAX_BATCH={"r_256_4_128_4":4,"r_128_16_4_16_4":4,"r_128_16_4_32_4":4,
           "r_8_16_4_8_4":4,"r_8_4_8_4":4,"r_8_4_8_4n1":4}
MAX_BATCH.update({"r_16_16_4_8_4":4,"r_4_16_4_8_4":4,"r_16_16_4_4":4,
                  "r_4_4_4_4":4,"r_16_16_4_4n1":4,"r_4_4_4_4n1":4})


def batched_source(source:str, name:str, batch_count:int) -> str:
  match=re.search(r"__kernel void \w+\((.*?)\) \{",source,re.S)
  if match is None: raise RuntimeError("kernel signature not found")
  declarations=[x.strip() for x in match.group(1).split(",")]
  arg_names=[x.rsplit(" ",1)[1] for x in declarations]
  renamed=[]
  bodies=[]
  body=source[match.end():source.rfind("}")]
  local_decls=re.findall(r"__attribute__\s*\(\(aligned \(\d+\)\)\)\s*__local\s+[^;]+;",body)
  hoisted=[]
  for batch in range(batch_count):
    mapping={arg:f"{arg}_{batch}" for arg in arg_names}
    renamed.extend(decl.rsplit(" ",1)[0]+" "+mapping[arg] for decl,arg in zip(declarations,arg_names))
    branch=body
    for declaration in local_decls:
      local_match=re.search(r"(\w+)(\[[^;]+;)$",declaration)
      if local_match is None: raise RuntimeError(f"local declaration not understood: {declaration}")
      old=local_match.group(1)
      new=f"{old}_{batch}"
      hoisted.append(declaration[:local_match.start(1)]+new+local_match.group(2))
      branch=branch.replace(declaration,"")
      branch=re.sub(rf"\b{re.escape(old)}\b",new,branch)
    for old,new in mapping.items(): branch=re.sub(rf"\b{re.escape(old)}\b",new,branch)
    bodies.append(branch)
  prefix=source[:match.start()]
  count=len(declarations)
  order=tuple(batch*count for batch in range(batch_count))+tuple(
    batch*count+arg for batch in range(batch_count) for arg in range(1,count))
  branches=" else ".join((f"if (get_group_id(1)=={batch}) " if batch < batch_count-1 else "")+f"{{{body}}}"
                         for batch,body in enumerate(bodies))
  return f"{prefix}__kernel void {name}_batch{batch_count}({','.join(renamed[i] for i in order)}) {{\n" \
         f"{''.join(hoisted)}\n{branches}\n}}"


def independent(calls:list) -> bool:
  outputs={call.src[out+1] for call in calls for out in call.src[0].arg.outs}
  return not any(arg in outputs for call in calls for i,arg in enumerate(call.src[1:]) if i not in call.src[0].arg.outs)


def batch_head(model) -> int:
  outer=model.captured.linear.src[0]
  batch=list(outer.src[0].src[0].src)
  new_batch=[]
  combined=0
  index=0
  cache={}
  while index < len(batch):
    first=batch[index]
    name=plain_name(first.src[0].arg.name) if first.op is Ops.CALL and first.src[0].op is Ops.PROGRAM else ""
    if index+1 < len(batch) and name in MAX_BATCH:
      calls=[first]
      while index+len(calls) < len(batch) and len(calls) < MAX_BATCH[name]:
        candidate=batch[index+len(calls)]
        candidate_name=plain_name(candidate.src[0].arg.name) if candidate.op is Ops.CALL and candidate.src[0].op is Ops.PROGRAM else ""
        if candidate_name != name or first.src[0].src[3].arg != candidate.src[0].src[3].arg: break
        calls.append(candidate)
      if len(calls) > 1 and independent(calls):
        batch_count=len(calls)
        program=first.src[0]
        source=batched_source(program.src[2].arg,name,batch_count)
        if source not in cache: cache[source]=Device["QCOM"].compiler.compile_cached(source)
        aux0=program.arg.aux[0]
        count=len(aux0)
        ordered_aux=tuple(aux0[0] for _ in calls)+tuple(entry for _ in calls for entry in aux0[1:])
        combined_aux=tuple(tuple((new_index,dtype,shape) for _old_index,dtype,shape in entry)
                           for new_index,entry in enumerate(ordered_aux))
        info=replace(program.arg,name=f"{name}_batch{batch_count}",global_size=(program.arg.global_size[0],batch_count,1),
                     globals=tuple(range(count*batch_count)),outs=tuple(range(batch_count)),
                     ins=tuple(range(batch_count,count*batch_count)),aux=(combined_aux,))
        program=program.replace(arg=info,src=program.src[:2]+
          (program.src[2].replace(arg=source),program.src[3].replace(arg=cache[source])))
        new_batch.append(first.replace(src=(program,*[call.src[1] for call in calls],
                                                *[arg for call in calls for arg in call.src[2:]])))
        combined+=1
        index+=batch_count
        continue
    new_batch.append(first)
    index+=1
  model.captured._linear=model.captured.linear.substitute({outer:create_graph_call(new_batch)},walk=True)
  model.captured.__dict__.pop("linear",None)
  return combined


def main() -> None:
  parser=argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  args=parser.parse_args()
  with open(args.input,"rb") as f:model=pickle.load(f)
  print("combined",batch_head(model))
  with open(args.output,"wb") as f:pickle.dump(model,f)


if __name__ == "__main__":main()
