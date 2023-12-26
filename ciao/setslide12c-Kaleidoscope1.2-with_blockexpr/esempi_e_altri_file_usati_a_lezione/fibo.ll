define double @fibo(double %n) {
entry:
  %f2 = alloca double, align 8
  %f1 = alloca double, align 8
  %n1 = alloca double, align 8
  store double %n, ptr %n1, align 8
  %n2 = load double, ptr %n1, align 8
  %lttest = fcmp ult double %n2, 2.000000e+00
  br i1 %lttest, label %trueexp, label %falseexp

trueexp:                                          ; preds = %entry
  br label %endcond

falseexp:                                         ; preds = %entry
  %n3 = load double, ptr %n1, align 8
  %subres = fsub double %n3, 1.000000e+00
  %calltmp = call double @fibo(double %subres)
  store double %calltmp, ptr %f1, align 8
  %n4 = load double, ptr %n1, align 8
  %subres5 = fsub double %n4, 2.000000e+00
  %calltmp6 = call double @fibo(double %subres5)
  store double %calltmp6, ptr %f2, align 8
  %f17 = load double, ptr %f1, align 8
  %f28 = load double, ptr %f2, align 8
  %addres = fadd double %f17, %f28
  br label %endcond

endcond:                                          ; preds = %falseexp, %trueexp
  %condval = phi double [ 1.000000e+00, %trueexp ], [ %addres, %falseexp ]
  ret double %condval
}

define double @ciao() {
entry:
  %n = load double, ptr %n1, align 8
  %addres = fadd double %n, 1.000000e+00
  ret double %addres
}

