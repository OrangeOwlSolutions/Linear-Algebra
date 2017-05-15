#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace af;

int main(int argc, char *argv[])
{
	const int N = 1000;

	try {

		// --- Select a device and display arrayfire info
		int device = argc > 1 ? atoi(argv[1]) : 0;
		af::setDevice(device);
		af::info();

		array A = randu(N, N, f64);
		af::array U, S, Vt;

		// --- Warning up
		timer time_last = timer::start();
		af::svd(U, S, Vt, A);
		S.eval();
		af::sync();
		double elapsed = timer::stop(time_last);
		printf("elapsed time using start and stop = %g ms \n", 1000.*elapsed);

		time_last = timer::start();
		af::svd(U, S, Vt, A);
		S.eval();
		af::sync();
		elapsed = timer::stop(time_last);
		printf("elapsed time using start and stop = %g ms \n", 1000.*elapsed);

	}
	catch (af::exception& e) {

		fprintf(stderr, "%s\n", e.what());
		throw;
	}

	return 0;
}
