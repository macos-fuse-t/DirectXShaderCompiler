target triple = "dxil-ms-dx"
define void @main() {
entry:
  %input = call half @dx.op.loadInput.f16(i32 4, i32 0, i32 0, i8 0, i32 undef)
  %sum = fadd half %input, 0xH3C00
  call void @dx.op.storeOutput.f16(i32 5, i32 0, i32 0, i8 0, half %sum)
  ret void
}
declare half @dx.op.loadInput.f16(i32, i32, i32, i8, i32) nounwind readnone
declare void @dx.op.storeOutput.f16(i32, i32, i32, i8, half) nounwind
!dx.version = !{!0}
!dx.valver = !{!0}
!dx.shaderModel = !{!1}
!dx.entryPoints = !{!2}
!0 = !{i32 1, i32 0}
!1 = !{!"ps", i32 6, i32 0}
!2 = !{void ()* @main, !"main", !3, null, !9}
!3 = !{!4, !7, null}
!4 = !{!5}
!5 = !{i32 0, !"TEXCOORD", i8 8, i8 0, !6, i8 2, i32 1, i8 1, i32 0, i8 0, null}
!6 = !{i32 0}
!7 = !{!8}
!8 = !{i32 0, !"SV_Target", i8 8, i8 16, !6, i8 0, i32 1, i8 1, i32 0, i8 0, null}
!9 = !{i32 0, i64 288}
