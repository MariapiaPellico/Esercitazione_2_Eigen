#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

bool solveSystemWithPALU(Matrix2d& A, Vector2d& b, Vector2d& x, double& errRel)
{
    JacobiSVD<Matrix2d> svd(A);
    Vector2d singularValuesA = svd.singularValues();
    if( singularValuesA.minCoeff() < 1e-16)
    {
        errRel = -1;
        return false;
    }

    Vector2d xPALU = A.fullPivLu().solve(b);
    errRel = (x-xPALU).norm() / x.norm();
    return true;

}

bool solveSystemWithQR(Matrix2d& A, Vector2d& b, Vector2d& x, double& errRel)
{
    JacobiSVD<Matrix2d> svd(A);
    Vector2d singularValuesA = svd.singularValues();
    if( singularValuesA.minCoeff() < 1e-16)
    {
        errRel = -1;
        return false;
    }

    Vector2d xQR = A.householderQr().solve(b);
    errRel = (x-xQR).norm() / x.norm();
    return true;

}

int main()
{
    //memorizziamo le matrici
    Matrix2d A_1, A_2, A_3;
    A_1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;

    A_2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;

    A_3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;

    //memorizziamo i vettori
    Vector2d b_1, b_2, b_3;
    b_1 << -5.169911863249772e-01, 1.672384680188350e-01;
    b_2 << -6.394645785530173e-04, 4.259549612877223e-04;
    b_3 << -6.400391328043042e-10, 4.266924591433963e-10;

    //memorizziamo la soluzione x dei tre sistemi lineari
    Vector2d x;
    x << -1.0e+0, -1.0e+00;

    double errRel1PALU, errRel2PALU, errRel3PALU;

    //risolviamo i sistemi con la fattorizzazione PALU
    if (solveSystemWithPALU(A_1,b_1,x,errRel1PALU))
        cout << scientific << "Il primo sistema lineare, risolto con la fattorizzazione PALU, ha un errore relativo pari a: " << errRel1PALU << endl;
    else
        cout << "La matrice A_1 risulta singolare" << endl;

    if (solveSystemWithPALU(A_2,b_2,x,errRel2PALU))
        cout << scientific << "Il secondo sistema lineare, risolto con la fattorizzazione PALU, ha un errore relativo pari a: " << errRel2PALU << endl;
    else
        cout << "La matrice A_2 risulta singolare" << endl;

    if (solveSystemWithPALU(A_3,b_3,x,errRel3PALU))
        cout << scientific << "Il terzo sistema lineare, risolto con la fattorizzazione PALU, ha un errore relativo pari a: " << errRel3PALU << endl;
    else
        cout << "La matrice A_3 risulta singolare" << endl;

    double errRel1QR, errRel2QR, errRel3QR;

    //risolviamo i sistemi con la fattorizzazione QR
    if (solveSystemWithQR(A_1,b_1,x,errRel1QR))
        cout << scientific << "Il primo sistema lineare, risolto con la fattorizzazione QR, ha un errore relativo pari a: " << errRel1QR << endl;
    else
        cout << "La matrice A_1 risulta singolare" << endl;

    if (solveSystemWithQR(A_2,b_2,x,errRel2QR))
        cout << scientific << "Il secondo sistema lineare, risolto con la fattorizzazione QR, ha un errore relativo pari a: " << errRel2QR << endl;
    else
        cout << "La matrice A_2 risulta singolare" << endl;

    if (solveSystemWithQR(A_3,b_3,x,errRel3QR))
        cout << scientific << "Il terzo sistema lineare, risolto con la fattorizzazione QR, ha un errore relativo pari a: " << errRel3QR << endl;
    else
        cout << "La matrice A_3 risulta singolare" << endl;

    return 0;
}
