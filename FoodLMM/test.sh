#CUDA_VISIBLE_DEVICES=3 python test_OOD_103.py --version="/home/qhy/FoodLISA/runs/ckpt_model/FoodLMM_S1/9500" --refer=1
#CUDA_VISIBLE_DEVICES=3 python test_OOD_103.py --version="/home/qhy/FoodLISA/runs/ckpt_model/FoodLMM_S1/9500" --refer=3
#
#CUDA_VISIBLE_DEVICES=3 python test_seg_103.py --version="/home/qhy/FoodLISA/runs/ckpt_model/FoodLMM_S1/9500" --refer=1
#CUDA_VISIBLE_DEVICES=3 python test_seg_103.py --version="/home/qhy/FoodLISA/runs/ckpt_model/FoodLMM_S1/9500" --refer=3
#CUDA_VISIBLE_DEVICES=3 python test_seg_103.py --version="/home/qhy/FoodLISA/runs/ckpt_model/FoodLMM_S1/9500" --refer=5
#
#CUDA_VISIBLE_DEVICES=3 python test_OOD_UEC.py --version="/home/qhy/FoodLISA/runs/ckpt_model/FoodLMM_S1/9500" --refer=1
#CUDA_VISIBLE_DEVICES=3 python test_OOD_UEC.py --version="/home/qhy/FoodLISA/runs/ckpt_model/FoodLMM_S1/9500" --refer=3
#
#CUDA_VISIBLE_DEVICES=3 python test_seg_UEC.py --version="/home/qhy/FoodLISA/runs/ckpt_model/FoodLMM_S1/9500" --refer=1
#CUDA_VISIBLE_DEVICES=3 python test_seg_UEC.py --version="/home/qhy/FoodLISA/runs/ckpt_model/FoodLMM_S1/9500" --refer=3
#CUDA_VISIBLE_DEVICES=3 python test_seg_UEC.py --version="/home/qhy/FoodLISA/runs/ckpt_model/FoodLMM_S1/9500" --refer=5
# CUDA_VISIBLE_DEVICES=0 python test_chinesefoodnet_title.py --version="/home/qhy/FoodLISA/runs/ckpt_model/FoodLMM_ft_chinesefoodnet/2000"

CUDA_VISIBLE_DEVICES=1 python test_dish_name.py