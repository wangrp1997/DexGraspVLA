#!/usr/bin/env python3
"""测试脚本：用于测试 DexGraspVLAPlanner 的各种功能."""

import os
import sys
import base64
import json
from PIL import Image, ImageDraw

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.dexgraspvla_planner import DexGraspVLAPlanner


def create_test_image(width=640, height=480, color=(100, 150, 200)):
    """创建测试图像."""
    # 创建一个简单的测试图像
    image = Image.new('RGB', (width, height), color)
    
    # 添加一些简单的图形来模拟物体
    draw = ImageDraw.Draw(image)
    
    # 画一个红色圆形（模拟杯子）
    draw.ellipse([100, 100, 200, 200], fill='red', 
                 outline='darkred', width=3)
    
    # 画一个蓝色矩形（模拟盒子）
    draw.rectangle([300, 150, 400, 250], fill='blue', 
                   outline='darkblue', width=3)
    
    # 画一个绿色三角形（模拟水果）
    draw.polygon([(500, 100), (550, 200), (450, 200)], 
                 fill='green', outline='darkgreen', width=3)
    
    return image


def image_to_base64(image):
    """将PIL图像转换为base64格式."""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{img_str}'


def test_planner_basic():
    """测试规划器的基本功能."""
    print('=' * 50)
    print('开始测试 DexGraspVLAPlanner')
    print('=' * 50)
    
    # 1. 创建规划器实例
    print('1. 创建规划器实例...')
    try:
        planner = DexGraspVLAPlanner(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name='qwen2.5-vl-72b-instruct'  # 千问多模态模型
        )
        print('✅ 规划器创建成功')
    except Exception as e:
        print(f'❌ 规划器创建失败: {e}')
        return False
    
    # 2. 设置日志
    print('\n2. 设置日志...')
    log_dir = 'test_logs'
    image_dir = 'test_images'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    log_file = open(os.path.join(log_dir, 'planner_test.log'), 'w')
    planner.set_logging(log_file, image_dir)
    print('✅ 日志设置完成')
    
    # 3. 创建测试图像
    print('\n3. 创建测试图像...')
    initial_image = create_test_image(color=(200, 200, 200))  # 浅灰色背景
    current_image = create_test_image(color=(180, 180, 180))  # 稍深的灰色背景
    
    # 转换为base64
    initial_image_b64 = image_to_base64(initial_image)
    current_image_b64 = image_to_base64(current_image)
    
    print('✅ 测试图像创建完成')
    
    # 4. 测试抓取指令生成
    print('\n4. 测试抓取指令生成...')
    try:
        vl_inputs = {
            'user_prompt': 'Pick up the red cup',
            'images': {
                'initial_head_image': initial_image_b64,
                'current_head_image': current_image_b64
            }
        }
        
        grasping_instruction = planner.request_task(
            task_name='grasping_instruction_proposal',
            vl_inputs=vl_inputs
        )
        print(f'✅ 抓取指令生成成功: {grasping_instruction}')
        
    except Exception as e:
        print(f'❌ 抓取指令生成失败: {e}')
        log_file.close()
        return False
    
    # 5. 测试边界框预测
    print('\n5. 测试边界框预测...')
    try:
        bbox_inputs = {
            'grasping_instruction': grasping_instruction,
            'images': {
                'current_head_image': current_image_b64
            }
        }
        
        bbox_result = planner.request_task(
            task_name='bounding_box_prediction',
            vl_inputs=bbox_inputs
        )
        print(f'✅ 边界框预测成功: {json.dumps(bbox_result, indent=2)}')
        
    except Exception as e:
        print(f'❌ 边界框预测失败: {e}')
        log_file.close()
        return False
    
    # 6. 测试抓取结果验证
    print('\n6. 测试抓取结果验证...')
    try:
        # 创建一个'抓取后'的图像（移除红色圆形）
        post_grasp_image = create_test_image(color=(180, 180, 180))
        draw = ImageDraw.Draw(post_grasp_image)
        # 只画蓝色和绿色物体，红色物体被'抓取'了
        draw.rectangle([300, 150, 400, 250], fill='blue', 
                       outline='darkblue', width=3)
        draw.polygon([(500, 100), (550, 200), (450, 200)], 
                     fill='green', outline='darkgreen', width=3)
        
        post_grasp_b64 = image_to_base64(post_grasp_image)
        
        verification_inputs = {
            'grasping_instruction': grasping_instruction,
            'images': {
                'current_head_image': post_grasp_b64,
                'current_wrist_image': post_grasp_b64  # 使用相同图像作为手腕图像
            }
        }
        
        verification_result = planner.request_task(
            task_name='grasp_outcome_verification',
            vl_inputs=verification_inputs
        )
        print(f'✅ 抓取结果验证成功: {verification_result}')
        
    except Exception as e:
        print(f'❌ 抓取结果验证失败: {e}')
        log_file.close()
        return False
    
    # 7. 测试任务完成检查
    print('\n7. 测试任务完成检查...')
    try:
        completion_inputs = {
            'user_prompt': 'Pick up the red cup',
            'images': {
                'initial_head_image': initial_image_b64,
                'current_head_image': post_grasp_b64
            }
        }
        
        completion_result = planner.request_task(
            task_name='prompt_completion_check',
            vl_inputs=completion_inputs
        )
        print(f'✅ 任务完成检查成功: {completion_result}')
        
    except Exception as e:
        print(f'❌ 任务完成检查失败: {e}')
        log_file.close()
        return False
    
    # 8. 保存测试图像
    print('\n8. 保存测试图像...')
    initial_image.save(os.path.join(image_dir, 'test_initial.png'))
    current_image.save(os.path.join(image_dir, 'test_current.png'))
    post_grasp_image.save(os.path.join(image_dir, 'test_post_grasp.png'))
    print('✅ 测试图像保存完成')
    
    # 9. 清理和总结
    log_file.close()
    
    print('\n' + '=' * 50)
    print('测试完成！')
    print('=' * 50)
    print('📁 日志文件: test_logs/planner_test.log')
    print('🖼️ 测试图像: test_images/')
    print('✅ 所有功能测试通过')
    
    return True


def test_planner_with_real_images():
    """使用真实图像测试（如果有的话）."""
    print('\n' + '=' * 50)
    print('测试真实图像（可选）')
    print('=' * 50)
    
    # 检查是否有真实图像
    real_image_path = 'test_images/real_test.png'
    if not os.path.exists(real_image_path):
        print('⚠️ 没有找到真实测试图像，跳过此测试')
        print('💡 提示：将你的测试图像命名为 real_test.png 放在 test_images/ 目录下')
        return
    
    try:
        # 加载真实图像
        real_image = Image.open(real_image_path)
        real_image_b64 = image_to_base64(real_image)
        
        # 创建规划器
        planner = DexGraspVLAPlanner(
            api_key='EMPTY',
            base_url='http://localhost:8000/v1',
            model_name=None
        )
        
        # 设置日志
        log_file = open('test_logs/real_image_test.log', 'w')
        planner.set_logging(log_file, 'test_images')
        
        # 测试真实图像
        vl_inputs = {
            'user_prompt': 'Identify objects in this image',
            'images': {
                'initial_head_image': real_image_b64,
                'current_head_image': real_image_b64
            }
        }
        
        result = planner.request_task(
            task_name='grasping_instruction_proposal',
            vl_inputs=vl_inputs
        )
        
        print(f'✅ 真实图像测试成功: {result}')
        log_file.close()
        
    except Exception as e:
        print(f'❌ 真实图像测试失败: {e}')


def main():
    """主函数."""
    print('DexGraspVLAPlanner 测试脚本')
    print('=' * 50)
    
    # 检查依赖
    try:
        import PIL
        print(f'✅ PIL版本: {PIL.__version__}')
    except ImportError:
        print('❌ 缺少PIL库，请安装: pip install Pillow')
        return
    
    # 运行基本测试
    success = test_planner_basic()
    
    if success:
        # 运行真实图像测试
        test_planner_with_real_images()
        
        print('\n🎉 所有测试完成！')
        print('\n📋 测试结果:')
        print('- 基本功能测试: ✅ 通过')
        print('- 图像处理: ✅ 正常')
        print('- API调用: ✅ 正常')
        print('- 日志记录: ✅ 正常')
        
        print('\n💡 使用提示:')
        print('1. 查看 test_logs/planner_test.log 了解详细日志')
        print('2. 查看 test_images/ 目录下的测试图像')
        print('3. 修改API密钥和模型配置以使用真实的LLM服务')
        
    else:
        print('\n❌ 测试失败，请检查配置和网络连接')


if __name__ == '__main__':
    main() 