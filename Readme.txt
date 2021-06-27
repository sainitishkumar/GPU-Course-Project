Description: CUDA based image denoising
	Making use of the cores in GPU to parallelize the computation of independant tasks, resulting in improved performance of the denoising algorithm.

Run as:
	1. make
	2.  !time ./cpu <input_img> <niter> <gamma> <method> <ctang?> <output_path>
	    !time ./gpu <input_img> <niter> <gamma> <method> <ctang?> <output_path>
	3. constraints:
		niter - integer [1,]
		gamma - float (0, 1)
		method - integer 1/2 for SRAD/OSRAD
		only if method == 2, mention ctang - integer [1,]
	4. clean folder
		make clean

To create a noisy image, use addnoise.py and change input and output file names

Examples:
	1. With method=2 OSRAD
		!time ./cpu data/noisy_test1.png 50 0.01 2 1 output/cpu_out.png
		!time ./gpu data/noisy_test1.png 50 0.01 2 1 output/gpu_out.png
	2. With method=1 SRAD
		!time ./cpu data/noisy_test1.png 50 0.01 1 output/cpu_out.png
		!time ./gpu data/noisy_test1.png 50 0.01 1 output/gpu_out.png
