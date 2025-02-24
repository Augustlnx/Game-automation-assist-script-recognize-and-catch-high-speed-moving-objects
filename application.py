import pygame
import time
import subprocess
import win32gui
import win32con
import pyautogui
import keyboard
import cv2
import numpy as np
from PIL import ImageGrab
import win32ui
from ctypes import windll

class GameController:
    def __init__(self):
        pygame.init()
        self.scrcpy_path = r"scrcpy-win64-v3.1\scrcpy.exe"
        self.hwnd = None
        self.window_width = None
        self.window_height = None
        self.lanes = []
        self.reflect_dist = 40
        self.template_params = {
            'tangyuan': {'ratio': 0.7*0.35, 'nms_threshold': 0.3, 'color': (103,69,152)},
            'bomb': {'ratio': 0.55*0.35, 'nms_threshold': 0.3, 'color': (0,0,0)},
            'shield': {'ratio': 0.55*0.35, 'nms_threshold': 0.3, 'color': (255,255,255)},
            'clock_minus': {'ratio': 0.7*0.35, 'nms_threshold': 0.3, 'color': (221,0,22)},
            'clock_plus': {'ratio': 0.7*0.35, 'nms_threshold': 0.3, 'color': (40,203,21)}
        }
        self.offset_table = {
        'clock_minus': {'x': 10, 'h': 45},
        'clock_plus': {'x': 10, 'h': 45},
        'bomb': {'x': 0, 'h': 8},
                'tangyuan': {'x': 0, 'h': 9}
        }
        
        # 加载原始篮子模板
        self.basket_template_original = cv2.imread('templates/basket.png')
        if self.basket_template_original is None:
            raise Exception("无法加载篮子模板图片")
        self.basket_template = None  # 将在init_window_params中调整大小
        print("成功加载篮子模板")

        # 加载模板图像
        self.templates = {}
        for name in ['tangyuan', 'bomb', 'clock_plus', 'clock_minus']:#, 'shield']:
            img = cv2.imread(f'templates/{name}.jpg')
            self.templates[name] = img

        self.resized_templates = {}  # 添加缓存字典
        # 预先调整所有模板大小
        for name, template in self.templates.items():
            params = self.template_params[name]
            resized = cv2.resize(template, 
                               (int(template.shape[1] * params['ratio']), 
                                int(template.shape[0] * params['ratio'])))
            self.resized_templates[name] = resized

        self._screen_cache = None
        self._last_capture_time = 0
        self._cache_timeout = 0.004  # 约120*2fps
        # 初始化截图相关的对象
        self._dc = None
        self._memdc = None
        self._bitmap = None
        self._bmp_info = None

    def get_game_window(self):
        """直接获取已知窗口句柄"""
        windows = []
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title == "JAD-AL00":
                # if title == 'HBN-ALOO':
                    windows.append(hwnd)
            return True
        
        try:
            win32gui.EnumWindows(callback, None)
            if windows:
                self.hwnd = windows[0]
                return True
        except Exception as e:
            print(f"查找窗口出错: {e}")
        return False

    def init_window_params(self):
        """初始化窗口参数"""
        try:
            # 获取窗口尺寸
            rect = win32gui.GetWindowRect(self.hwnd)
            if not rect:
                return False
                
            left, top, right, bottom = rect
            self.window_width = right - left
            self.window_height = bottom - top
            
            # 调整篮子模板大小为窗口宽度的1/4
            target_width = int(self.window_width // 4 * 1.1)
            aspect_ratio = self.basket_template_original.shape[1] / self.basket_template_original.shape[0]
            target_height = int(target_width / aspect_ratio)
            self.basket_template = cv2.resize(self.basket_template_original, (target_width, target_height))
            print(f"调整后的模板大小: {target_width}x{target_height}")
            
            # 计算5个轨道位置
            lane_width = self.window_width / 5
            self.lanes = [int(lane_width * (i + 0.5)) for i in range(5)]
            print(f"窗口尺寸: {self.window_width}x{self.window_height}")
            print(f"轨道位置: {self.lanes}")
            return True
        except Exception as e:
            print(f"初始化参数失败: {e}")
            return False

    def _init_screen_capture(self):
        """初始化截图需要的DC和位图对象"""
        try:
            # 获取窗口DC
            self._dc = win32gui.GetWindowDC(self.hwnd)
            self._memdc = win32ui.CreateDCFromHandle(self._dc)
            self._save_dc = self._memdc.CreateCompatibleDC()
            
            # 创建位图
            self._bitmap = win32ui.CreateBitmap()
            self._bitmap.CreateCompatibleBitmap(self._memdc, self.window_width, self.window_height)
            self._save_dc.SelectObject(self._bitmap)
            
            return True
        except Exception as e:
            print(f"初始化截图对象失败: {e}")
            self._release_screen_capture()
            return False
    
    def _release_screen_capture(self):
        """释放截图相关资源"""
        try:
            if self._bitmap:
                win32gui.DeleteObject(self._bitmap.GetHandle())
            if self._save_dc:
                self._save_dc.DeleteDC()
            if self._memdc:
                self._memdc.DeleteDC()
            if self._dc:
                win32gui.ReleaseDC(self.hwnd, self._dc)
        except Exception as e:
            print(f"释放截图资源出错: {e}")
        finally:
            self._dc = None
            self._memdc = None
            self._save_dc = None
            self._bitmap = None

    def get_screen(self):
        """获取窗口截图 - 优化版本"""
        if not self.hwnd:
            return None
            
        try:
            current_time = time.time()
            
            # 使用缓存
            if (self._screen_cache is not None and 
                current_time - self._last_capture_time < self._cache_timeout):
                return self._screen_cache.copy()
            
            # 初始化或重新初始化截图对象
            if self._dc is None and not self._init_screen_capture():
                return None
            
            # 执行截图
            self._save_dc.BitBlt(
                (0, 0), 
                (self.window_width, self.window_height), 
                self._memdc, 
                (0, 0), 
                win32con.SRCCOPY
            )
            
            # 获取位图数据
            bmp_str = self._bitmap.GetBitmapBits(True)
            img = np.frombuffer(bmp_str, dtype=np.uint8).reshape(
                self.window_height, 
                self.window_width, 
                4
            )
            
            # 转换颜色空间并更新缓存
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self._screen_cache = img.copy()
            self._last_capture_time = current_time
            
            return img
            
        except Exception as e:
            print(f"截图错误: {e}")
            self._release_screen_capture()  # 出错时释放资源
            return None

    def get_basket_position(self, screen, show = False):
        """获取篮子位置"""
        try:
            if self.basket_template is None:
                #print("篮子模板未初始化")
                return None
                
            # 在屏幕下半部分搜索以提高效率和准确性
            begin_y = screen.shape[0] // 2 + 300
            search_region = screen[begin_y:, :]
            
            # 进行模板匹配
            result = cv2.matchTemplate(search_region, self.basket_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # 调试信息
            #print(f"模板匹配最大值: {max_val:.2f}")
            
            if max_val >= 0.35:  # 降低阈值以适应实际情况
                # 计算实际坐标（考虑搜索区域的偏移）
                center_x = max_loc[0] + self.basket_template.shape[1] // 2
                center_y = begin_y + max_loc[1] + self.basket_template.shape[0] // 2
                
                upper_edge_y = center_y - self.basket_template.shape[0]//2
                if show:
                    # 可视化匹配结果（调试用）
                    debug_screen = screen.copy()
                    cv2.rectangle(debug_screen, 
                                (max_loc[0], begin_y + max_loc[1]),
                                (max_loc[0] + self.basket_template.shape[1], 
                                begin_y + max_loc[1] + self.basket_template.shape[0]),
                                (0, 255, 0), 2)
                    # 绘制上沿的蓝色参考虚线
                    
                    cv2.line(debug_screen, (0, upper_edge_y), (self.window_width, upper_edge_y), (255, 0, 0), 1)
                    cv2.imshow('Match Result', debug_screen)
                    cv2.waitKey(1)
                
                #print(f"找到篮子位置: ({center_x}, {center_y}), 匹配度: {max_val:.2f}")
                # 额外放回上沿位置
                return center_x, center_y, upper_edge_y
            else:
                #print(f"未找到篮子，最佳匹配度: {max_val:.2f}")
                return None
        except Exception as e:
            #print(f"获取篮子位置失败: {e}")
            return None
        
    def py_nms(self, dets, thresh):  # 添加self参数
        """Pure Python NMS implementation."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
    
    def get_item_position(self, screen, y_ref, show=False, begin = 240,end = 90,num_check=2):
        """获取所有物件位置"""
        items = []
        x_list = []
        debug_screen = screen.copy() if show else None
        # 在屏幕下半部分搜索以提高效率和准确性
        begin_y = y_ref - begin
        end_y = y_ref - end
        search_region = screen[begin_y:end_y, :]
        sum_items = 0
        t1 = time.time()
        for name, template in self.resized_templates.items():
            if sum_items >= num_check:
                break
            params = self.template_params[name]
            # resized_template = self.resized_templates[name]
            # # 调整模板大小
            # resized_template = cv2.resize(template, 
            #                             (int(template.shape[1] * params['ratio']), 
            #                              int(template.shape[0] * params['ratio'])))
            #t1 = time.time()
            # 执行模板匹配
            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            
            threshold = 0.8 if name == 'clock_plus' else 0.8
            if name == 'tangyuan':
                threshold = 0.7
            if name =='bomb':
                threshold = 0.9
            locations = np.where(result >= threshold)
            # scores = result[result >= 0.3]
            #t2 = time.time()
            #print(f"模板匹配耗时: {t2 - t1:.2f}秒")

            if len(locations[0]) > 0:
                w, h = template.shape[1], template.shape[0]
                # 构建基础坐标
                x_coords = locations[1]  # 获取x坐标
                y_coords = locations[0] + begin_y  # 获取y坐标并调整偏移
                scores = result[locations]  # 这样提取的scores维度会与locations匹配
                # 根据不同物品类型调整坐标
                # 使用查找表替代多重if条件

                
                offsets = self.offset_table.get(name, {'x': 0, 'h': 0})
                x_offset = offsets['x']
                h_offset = offsets['h']
                # # 验证数组维度
                # assert len(x_coords) == len(scores), "坐标和分数数量不匹配"
                                    
                # 使用向量化操作构建boxes
                boxes = np.column_stack([
                    x_coords + x_offset,  # x1
                    y_coords,            # y1
                    x_coords + w + x_offset,  # x2
                    y_coords + h + h_offset,  # y2
                    scores              # 分数
                ])
                
                if len(boxes) > 0:
                    boxes = np.array(boxes)
                    keep = self.py_nms(boxes, params['nms_threshold'])
                    filtered_boxes = boxes[keep]
                    sum_items += len(filtered_boxes)
                    if len(filtered_boxes) > 2 and name == 'clock_minus':
                        # 删掉item中的所有name为bomb的字典
                        items = [item for item in items if item['name'] != 'bomb']
                        x_list = [item['center'][0] for item in items]
                        # 如果items为空，则返回None，否则返回items和x_list
                        
                    for box in filtered_boxes:
                        center_x = int((box[0] + box[2]) // 2)
                        center_y = int((box[1] + box[3]) // 2)
                        y_bottom = int(box[3])
                        y_abs = abs(y_bottom - y_ref)
                        
                        items.append({
                            'name': name,
                            'center': (center_x, center_y),
                            'y_abs': y_abs,
                            'distance': y_bottom - y_ref
                        })
                        x_list.append(center_x)
                        
                        if show:
                            cv2.rectangle(debug_screen, 
                                        (int(box[0]), int(box[1])), 
                                        (int(box[2]), int(box[3])), 
                                        params['color'], 2)
                            cv2.putText(debug_screen,
                                      f"{name}:{y_abs:.0f}",
                                      (int(box[0]), int(box[1] - 10)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, params['color'], 2)
        # t2 = time.time()
        # print(f"模板匹配耗时: {t2 - t1:.2f}秒")
        if show and debug_screen is not None:
            cv2.line(debug_screen, (0, y_ref), (self.window_width, y_ref), (0, 255, 0), 2)
            cv2.line(debug_screen, (0, y_ref - self.reflect_dist), (self.window_width, y_ref - self.reflect_dist), (80, 160, 250), 2)
            cv2.rectangle(debug_screen,
                        (0, begin_y),
                        (self.window_width, end_y),
                        (0, 0, 255), 2)
            if items:
                min_item = min(items, key=lambda x: x['y_abs'])
                center_x, center_y = min_item['center']
                cv2.rectangle(debug_screen,
                    (center_x - 25, center_y - 25),
                    (center_x + 25, center_y + 25),
                    (0, 0, 255), 2)
                cv2.putText(debug_screen,
                       f"min_y_abs: {min_item['y_abs']:.0f}",
                       (center_x - 40, center_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('Items Detected', debug_screen)
            cv2.waitKey(1)

        if items:
            closest_item = min(items, key=lambda x: x['y_abs'])
            return closest_item['name'], closest_item['center'], closest_item['distance'], x_list
        
        return None, None, None, x_list

    def run(self):

        # 启动scrcpy
        print("启动scrcpy...")
        subprocess.Popen([self.scrcpy_path])
        time.sleep(3)
        pyautogui.PAUSE = 0.0001
        # 获取窗口
        if not self.get_game_window() or not self.init_window_params():
            print("无法找到或初始化窗口")
            return

        print(f"找到窗口，开始运行...")
        
        try:
            # 获取窗口位置
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            
            # 获取初始屏幕截图
            screen = self.get_screen()
            if screen is None:
                #print("无法获取屏幕截图")
                return
                
            # 获取篮子位置
            basket_pos = self.get_basket_position(screen)
            if basket_pos is None:
                #print("无法找到篮子位置，使用默认位置")
                center_x = left + self.window_width // 2
                center_y = top + self.window_height - 100
            else:
                center_x = left + basket_pos[0]
                center_y = top + basket_pos[1]
            
            #print(f"初始位置设置为: ({center_x}, {center_y})")
            current_x = center_x
    
            # 确保鼠标状态正确
            pyautogui.mouseUp()
            #time.sleep(0.5)
            
            # 开始移动
            pyautogui.moveTo(center_x, center_y)
            #time.sleep(0.5)
            pyautogui.mouseDown()
            
            # 控制循环
            # current_lane = 0
            # direction = 1
            current_pos = (current_x-left, center_y-top, basket_pos[2])
            begin, end = 220, 85
            num_check = 2
            start_time = time.time()
            while True:
                t1 = time.time()
                screen = self.get_screen()
                tt1 = time.time()  
                # basket_pos = self.get_basket_position(screen, show=True)
                basket_pos = current_pos
                if basket_pos is None:
                    # print("未检测到篮子位置")
                    continue
                  
                result = self.get_item_position(screen, basket_pos[2], show=False,begin=begin, end = end,num_check=num_check)
                # print(basket_pos[2]) 889
                if result[0] is None or result[1] is None:
                    # print("未检测到物品")
                    continue
                tt2 = time.time()
                # print(f"截屏耗时: {tt1-t1:.3f}s,总耗时: {tt2-t1:.3f}s")
                name_closest_item, (item_x, item_y), dist_item, x_list = result
                # print(name_closest_item,item_x)
                center_x = left + basket_pos[0]
                center_y = top + basket_pos[1]
                t2 = time.time()
                try:
                    #t4 = time.time()
                    #pyautogui.moveTo(center_x, center_y)
                    #t5 = time.time()
                    #pyautogui.mouseDown()
                    #t6 = time.time()
                    times = time.time() - start_time
                    if (times%30<0.05):
                        print(f'游戏时间:{times}')
                    self.reflect_dist = 90 if times < 100 else 100
                    num_check = 2 if times < 60 else 4
                    if times > 120:
                        num_check = 5
                        begin, end = 240, 115
                        self.reflect_dist = 120
                    if times > 150:
                        # num_check = 5
                        begin, end = 260, 115
                        self.reflect_dist = 140
                    if times > 180:
                        num_check = 4
                        begin, end = 270, 115
                        self.reflect_dist = 150
                    if times > 220:
                        # num_check = 2
                        begin, end = 280, 130
                        self.reflect_dist = 160
                    if times > 250:
                        begin, end = int(290 + (times-250) / 30*20), 130
                        self.reflect_dist = 170 + (times-250) / 30*20
                    if times > 280:
                        num_check = 4
                        begin, end = int(320 + (times-270) / 50*20), int(130 + (times-270) / 50*20)
                        self.reflect_dist = 190 + (times-270) / 50*20
                    if times > 330:
                        begin, end = int(340 + (times-330) / 30*20), int(130 + (times-270) / 50*20)
                        self.reflect_dist = 210 + (times-330) / 30*20
                    if times > 530:
                        begin, end = int(513 + (times-560) / 15*20), int(130 + (times-270) / 50*20)
                        self.reflect_dist = 370 + (times-560) / 15*20

                    #     self.offset_table['clock_minus']['h'] = 70
                    #     self.offset_table['clock_plus']['h'] = 70
                    # print(dist_item,self.reflect_dist)
                    if abs(dist_item) < self.reflect_dist:# + 15/20 * (times+5) :
                        
                        item_screen_x = left + item_x
                        if name_closest_item in ['tangyuan', 'shield','clock_plus']:
                            target_x = item_screen_x  # 加上窗口左边距
                        else:
                            # 如果current_x和x_closest_item的距离大于200，则无需移动，否则移动到x_list中距离x_closest_item最远处
                            if abs(current_x - item_screen_x) > 90:
                                continue
                            else:
                                # 找到离物品x坐标最远的边界
                                # lane = [70, 140, 210, 280, 350, 420]
                                # abs_lane = [abs(x - item_x) for x in x_list]
                                # min_index = abs_lane.index(min(abs_lane))
                                # lane.remove(lane[min_index])
                                # abs_lane = [abs(x - item_x) for x in x_list]
                                # min_index = abs_lane.index(min(abs_lane))
                                # target_x = left + lane[min_index]
                                # left_edge =   left + 70 # 左边界，留50像素边距
                                # right_edge =  left + 420 # 右边界，留50像素边距
                                target_x = left + 70 if item_x > 245 else left + 420
                    else: 
                        continue               
                    #time.sleep(0.002)
                    if target_x is not None:
                        pyautogui.moveTo(target_x, center_y, duration=0)
                    #t7 = time.time()
                    # pyautogui.mouseUp()
                    #t8 = time.time()
                        current_x = target_x
                        current_pos = (current_x-left, center_y-top, basket_pos[2])
                    
                except Exception as e:
                    print(f"移动操作出错: {e}")
                    pyautogui.mouseUp()
                    continue
                # t3 = time.time()
                # print(f"循环总耗时: {t3-t1:.3f}s,识别耗时: {t2-t1:.3f}s,移动耗时: {t3-t2:.3f}s")
                #print(f'鼠标移动耗时: {t5-t4:.3f}s,鼠标抬起耗时: {t8-t7:.3f}s,鼠标落下耗时: {t6-t5:.3f}s,鼠标按下耗时: {t6-t5:.3f}s')
                # 检查退出条件 - 改用keyboard库检测Esc键
                if not win32gui.IsWindow(self.hwnd):
                    break
                if keyboard.is_pressed('esc'):  # 改用Esc键退出
                    print("用户按下Esc键退出")
                    break
                    
        except Exception as e:
            import traceback
            print(f"运行出错: {e}\n详细信息:\n{traceback.format_exc()}")
            pyautogui.mouseUp()
        finally:
            self._release_screen_capture()  # 确保在程序结束时释放资源
            pyautogui.mouseUp()

def main():
    controller = GameController()
    controller.run()

if __name__ == "__main__":
    main()