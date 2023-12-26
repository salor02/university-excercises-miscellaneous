define double @a(double %n) {
entry:
  %n1 = alloca double, align 8
  store double %n, ptr %n1, align 8
  %n2 = load double, ptr %n1, align 8
  %addres = fadd double %n2, 1.000000e+00
  ret double %addres
}

define double @b(double %s) {
entry:
  %s1 = alloca double, align 8
  store double %s, ptr %s1, align 8
  %s2 = load double, ptr %s1, align 8
  %s3 = load double, ptr %s1, align 8
  %calltmp = call double @a(double %s3)
  %addres = fadd double %s2, %calltmp
  ret double %addres
}

