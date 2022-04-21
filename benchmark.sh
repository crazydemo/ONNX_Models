#!/bin/bash -x

target='llvm -mcpu=cascadelake -model=platinum-8280'
dtype='float32'

record_root="/home2/zhangya9/onnx_models/Inception_V3/torch_model/openvino"
file_name="inception_v3"
xml_pth="${record_root}/${file_name}.xml"
bin_pth="${record_root}/${file_name}.bin"
mapping_pth="${record_root}/${file_name}.mapping"

cores_list=('4') #('28' '28' '4')    # multi-instances should be the last one
batch_list=('1') #('1' '128' '1')

repeat=120
physical_cores=28

for i in $(seq 1 ${#cores_list[@]}); do
    num_cores=${cores_list[$i-1]}
    batch_size=${batch_list[$i-1]}
    num_groups=$[${physical_cores}/${num_cores}]

    export OMP_NUM_THREADS=${num_cores}
    export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
    for j in $(seq 0 $[${num_groups}-1]); do
        start_core=$[${j}*${num_cores}]
        end_core=$[$[${j}+1]*${num_cores}-1]
        benchmark_log="${record_root}/${num_cores}cores_bs${batch_size}_cores${start_core}-${end_core}.log"
        # benchmark_log="${record_root}/onednn.log"
        printf "=%.0s" {1..100}; echo
        echo "benchmarking autoscheduler with ${num_cores} cores and batchsize=${batch_size} on cores: ${start_core}-${end_core}"
        echo "saving logs to ${benchmark_log}"; echo
        
        if [ ${num_groups} == 1 ]
        then
            numactl --physcpubind=${start_core}-${end_core} --membind=0 \
            benchmark_app -m ${xml_pth} -d CPU -api sync -progress -b ${batch_size} -niter ${repeat}\
                | tee ${benchmark_log} 2>&1
            echo "done benchmarking autoscheduler with ${num_cores} cores and batchsize=${batch_size} on cores: ${start_core}-${end_core}"
        else
            numactl --physcpubind=${start_core}-${end_core} --membind=0 \
            benchmark_app -m ${xml_pth} -d CPU -api sync -progress -b ${batch_size} -niter ${repeat}\
                | tee ${benchmark_log} 2>&1 &
        fi
    done
done
echo "benchmarking sessions lanched, please wait for the python runs."

