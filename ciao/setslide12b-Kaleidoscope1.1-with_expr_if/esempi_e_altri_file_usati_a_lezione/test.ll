define double @a(double %n) {
entry:
  %n1 = alloca double, align 8
  store double %n, ptr %n1, align 8
  %n2 = load double, ptr %n1, align 8
  %addres = fadd double %n2, 2.000000e+00
  ret double %addres
}

Variabile non definita
