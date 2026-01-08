#include<iostream>
#include<cmath>
using namespace std;
int main(){


    long long m, seed, round;
    cin >> m >> seed >> round;

    while(round --){
        long long square = seed * seed;

        long long persudo = (square / (long long)pow(10, m / 2)) % (long long)pow(10,m);

        cout << persudo << endl;

        seed = persudo;

    }


    return 0;
}