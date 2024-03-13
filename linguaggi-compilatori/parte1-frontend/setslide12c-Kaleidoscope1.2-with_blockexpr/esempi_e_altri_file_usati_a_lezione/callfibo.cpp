#include <iostream>

extern "C" {
    double fibo(double);
}

extern "C" {
    double ciao();
}

int main() {
    int n;
    std::cout << "Inserisci il valore di n: ";
    std::cin >> n;
    std::cout << "fib(" << n << ") = " << fibo(n) << std::endl;
    std::cout<<ciao()<<std::endl;
}
