CUDA_VISIBLE_DEVICES=1 python test.py --config configs/webcari_munit.yaml --content_dir ../PytorchNeuralStyleTransfer/images/contents_total_0425/ --style_dir ../PytorchNeuralStyleTransfer/images/styles_0420 --output_folder 0621_SAMUNIT --checkpoint outputs/webcari_munit/checkpoints/gen_00230000.pt --trainer MUNIT

