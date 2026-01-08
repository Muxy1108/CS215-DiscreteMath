#include<iostream>
#include<cmath>
using namespace std;
int main(){

    long long m, a, c, x, round;
    cin >> m >> a >> c >> x >> round;

    while(round --){

        long long persudo = (a * x + c) % m;

        cout << persudo << endl;

        x = persudo;

    }

    return 0;
}