__kernel void sendEmployedBees(
	__global float *foods,
	__global float *trueFit,
	__global float *fitness,
	__global int *trial,
	__global float *rand,
	int gen){

	int i = get_global_id(0);

	int param2change = (int)(rand[gen * 64 + i] + 1);
	int k = (int)(rand[gen * 64 + i] * 64);
	k = (i + k) % 64;
	float arr[2];
	float temp_truefit = 0;
	arr[0] = foods[2 * i];
	arr[1] = foods[2 * i + 1];
	float Rij = rand[gen * 64 + i];
	arr[param2change] = foods[2 * i + param2change] + Rij * (foods[2 * i + param2change]-foods[2 * k + param2change]);  //根据公式(2-3)
	if(arr[param2change] > 100) {
		arr[param2change] = 100;
	}
	if(arr[param2change] < -100) {
		arr[param2change] = -100;
	}
	temp_truefit = 0.5 + (sin(sqrt(arr[0] * arr[0] + arr[1] * arr[1])) * sin(sqrt(arr[0] * arr[0] + arr[1] * arr[1])) - 0.5)
        / ((1 + 0.001 * (arr[0] * arr[0] + arr[1] * arr[1])) * (1 + 0.001 * (arr[0] * arr[0] + arr[1] * arr[1])));

	if(temp_truefit < trueFit[i]) {
		foods[2 * i] = arr[0];
		foods[2 * i + 1] = arr[1];
		trueFit[i] = temp_truefit;
		if(temp_truefit >= 0) {
			fitness[i] = 1 / temp_truefit;
		} else {
			fitness[i] = 1 - temp_truefit;
		}
		trial[i] = 0;
	} else {
		trial[i]++;
	}

}

__kernel void CalculateProbabilities(
	__global float *fitness,
	__global float *prob,
	__local float *localArray){
	float maxfit;
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	localArray[lid] = fitness[gid];
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
		if(lid < stride) {
			if(localArray[lid] < localArray[lid + stride]) {
				localArray[lid] = localArray[lid + stride];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(lid == 0) {
		maxfit = localArray[0];
	}
	prob[gid] =(0.9 * (fitness[gid] / maxfit))+0.1;	
}

__kernel void sendOnlookerBees(
	__global float *foods,
	__global float *trueFit,
	__global float *fitness,
	__global float *prob,
	__global int *trial,
	__global float *rand_1,
	__global float *rand_2,
	int gen){
	int index = get_global_id(0);
	float R_1 = rand_1[gen * 64 * 3 + index];
	float R_2 = rand_1[gen * 64 * 3 + index + 1];
	float R_3 = rand_1[gen * 64 * 3 + index + 2];
	int i = 0;
	if(prob[index] >= R_1) {
		i = index;
	} else if(prob[(index + 1) % 64] >= R_2) {
		i = (index + 1) % 64;
	} else if(prob[(index - 1) % 64] >= R_3) {
		i = (index - 1) % 64;
	} else {
		trial[index]++;
		return;
	}
	int param2change = (int)(rand_2[gen * 64 + i] + 1);
	int k = (int)(rand_2[gen * 64 + i] * 64);
	k = (i + k) % 64;
	float arr[2];
	arr[0] = foods[2 * i];
	arr[1] = foods[2 * i + 1];
	float temp_truefit = 0;
	float Rij = rand_2[gen * 64 + index];
	arr[param2change] = foods[2 * i + param2change] + Rij * (foods[2 * i + param2change]-foods[2 * k + param2change]);  //根据公式(2-3)
	if(arr[param2change] > 100) {
		arr[param2change] = 100;
	}
	if(arr[param2change] < -100) {
		arr[param2change] = -100;
	}
	temp_truefit = 0.5 + (sin(sqrt(arr[0] * arr[0] + arr[1] * arr[1])) * sin(sqrt(arr[0] * arr[0] + arr[1] * arr[1])) - 0.5)
        / ((1 + 0.001 * (arr[0] * arr[0] + arr[1] * arr[1])) * (1 + 0.001 * (arr[0] * arr[0] + arr[1] * arr[1])));

	if(temp_truefit < trueFit[i]) {
		foods[2 * i] = arr[0];
		foods[2 * i + 1] = arr[1];
		trueFit[i] = temp_truefit;
		if(temp_truefit >= 0) {
			fitness[i] = 1 / temp_truefit;
		} else {
			fitness[i] = 1 - temp_truefit;
		}
		trial[i] = 0;
	} else {
		trial[i]++;
	}
}

__kernel void MemorizeBestSource(
	__global float *trueFit,
	__global float *result,
	__local float *localArray){
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	localArray[lid] = trueFit[gid];
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
		if(lid < stride) {
			if(localArray[lid] > localArray[lid + stride]) {
				localArray[lid] = localArray[lid + stride];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (lid == 0) {
		// 选择一个工作项将局部最小值写回到全局内存
		result[0] = min(result[0], localArray[0]);
	}
}

__kernel void sendScoutBees(
	__global float *foods,
	__global float *trueFit,
	__global float *fitness,
	__global int *trial,
	__global float *rand,
	int gen){
	int i = get_global_id(0);
	float R;
	if(trial[i] >= 20) {
		for (int j = 0; j < 2;j++)
        {
            R = rand[gen * 64 * 2 + i + j];
            foods[2 * i + j] = -100 + R * 200;  
        }
        trial[i] = 0;
		float x = foods[2 * i];
		float y = foods[2 * i + 1];
        trueFit[i] = 0.5 + (sin(sqrt(x * x + y * y)) * sin(sqrt(x * x + y * y)) - 0.5)
        / ((1 + 0.001 * (x * x + y * y)) * (1 + 0.001 * (x * x + y * y)));
		if (trueFit[i] >= 0){
			fitness[i] = 1 / (trueFit[i] + 1);
        }
        else {
			fitness[i] = 1 - trueFit[i];
        }
	}
}