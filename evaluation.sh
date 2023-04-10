#!/use/bin/env bash
GPUid=0
runs_file=Apr08_18-24-20_j6fe5b6b93634061b7fd01e16b118580-task0-0-GPU0
echo $runs_file
Model_=DecorNet
python evaluation_test.py --model_path ./DecorNet/runs/$runs_file --GPU_id $GPUid --is_best_3D True
python Compute_Metric_test.py --output_path ./DecorNet/runs/$runs_file/output_test
