#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/13 11:38
# @Author  : h1code2
# @File    : common_onnx.py
# @Software: PyCharm

import json
import os
import time

import cv2
import numpy as np
import onnxruntime


class OnnxUtils:
    output_names = None
    input_width = None
    input_height = None
    input_names = None

    def __init__(self, model_path: str, class_names, conf_threes=0.5, iou_threes=0.5):
        # 初始化类参数并加载ONNX模型
        self.class_names = class_names
        self.conf_threshold = conf_threes
        self.iou_threshold = iou_threes
        self.colors = self.generate_colors(len(class_names))
        self.session = onnxruntime.InferenceSession(
            model_path, providers=onnxruntime.get_available_providers()
        )
        self.prepare_model_details()

    @staticmethod
    def generate_colors(num_classes):
        # 为每个类生成一个随机颜色
        rng = np.random.default_rng(3)
        return rng.uniform(0, 255, size=(num_classes, 3))

    def detect_and_draw(self, image, min_score: float = 0.8, cut: bool = False, rim: int = 10):
        """
        检测对象并绘制结果
        Args:
            image:
            min_score: 置信度
            cut: 是否裁剪
            rim: 裁剪空出的边框大小
        Returns:
        """
        det_image = image.copy()
        cut_image = image.copy()
        boxes, scores, class_ids = self.detect_objects(image)
        det_image = self.draw_masks(det_image, boxes, class_ids)
        results = []
        index = 0
        for class_id, box, score in zip(class_ids, boxes, scores):
            if score < min_score:
                continue
            color = self.colors[class_id]
            self.draw_box(det_image, box, color)
            # caption = f"{self.class_names[class_id]} {int(score * 100)}%"
            caption = "{:.2f}".format(score)
            # print("class_id:{} box:{} score:{:.2f}".format(class_id, box, score))
            if cut:
                from pathlib import Path
                Path("./small_image").mkdir(parents=True, exist_ok=True)
            small_image = cut_image[int(box[1] - rim): int(box[3] + rim), int(box[0]) - rim: int(box[2] + rim)]
            cv2.imwrite(os.path.join("./small_image/", f"cut_image_{index}.png"), small_image)
            results.append(
                {
                    "class_id": class_id,
                    "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
                    "score": caption,
                }
            )
            index += 1
            self.draw_text(det_image, caption, box, color)
        return results, det_image

    def detect_objects(self, image):
        # 检测图像中的对象
        input_tensor = self.prepare_input(image)
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )
        return self.process_output(outputs, image.shape[:2])

    def prepare_input(self, image):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img.transpose(2, 0, 1) / 255.0
        return input_img[np.newaxis, :].astype(np.float32)

    def process_output(self, outputs, image_shape):
        predictions = np.squeeze(outputs[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        indices = scores > self.conf_threshold
        predictions = predictions[indices]
        scores = scores[indices]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions[:, :4], image_shape)
        indices = self.non_maximum_suppression(boxes, scores, class_ids)
        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, boxes, image_shape):
        # 从模型预测中提取盒子并调整尺寸
        scale = np.array([self.input_width, self.input_height] * 2)
        boxes = (boxes / scale) * np.array([image_shape[1], image_shape[0]] * 2)
        return self.xywh_to_xyxy(boxes)

    @staticmethod
    def xywh_to_xyxy(boxes):
        # 转换框的坐标
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        return boxes

    def non_maximum_suppression(self, boxes, scores, class_ids):
        # 执行非最大抑制
        unique_class_ids = np.unique(class_ids)
        keep_boxes = []
        for class_id in unique_class_ids:
            indices = class_ids == class_id
            selected_boxes = boxes[indices]
            selected_scores = scores[indices]
            keep_indices = self.single_class_nms(selected_boxes, selected_scores)
            keep_boxes.extend(np.where(indices)[0][keep_indices])
        return np.array(keep_boxes, dtype=int)

    def single_class_nms(self, boxes, scores):
        # 执行单类非最大抑制
        if len(scores) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        # 迭代删除重叠框
        while order.size > 0:
            i = order[0]
            keep.append(i)
            intersections = np.maximum(
                0,
                np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1,
            ) * np.maximum(
                0,
                np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1,
            )
            iou = intersections / (areas[i] + areas[order[1:]] - intersections)
            order = order[np.where(iou <= self.iou_threshold)[0] + 1]
        return keep

    @staticmethod
    def compute_iou(box, boxes):
        # 计算交并比
        intersection = np.maximum(
            0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])
        )
        intersection_area = intersection[:, 0] * intersection[:, 1]
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return intersection_area / (box_area + boxes_area - intersection_area)

    @staticmethod
    def draw_box(image, box, color, thickness=1):
        # 绘制边框
        box = box.astype(int)
        return cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, thickness)

    @staticmethod
    def draw_text(image, text, box, color, font_size=0.5, text_thickness=1):
        """
        Draw text on
        Args:
            image:
            text:
            box:
            color: 字体颜色
            font_size: 字体大小
            text_thickness: 粗细
        Returns:
        """
        box = box.astype(int)
        position = (box[2] + 3, box[3])
        return cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            color,
            text_thickness,
            lineType=cv2.LINE_AA,  # 添加抗锯齿，以尽量减少锯齿效果
        )

    def draw_masks(self, image, boxes, classes):
        # 在图像上绘制遮罩
        mask_img = image.copy()
        for box, class_id in zip(boxes, classes):
            color = self.colors[class_id]
            cv2.rectangle(
                mask_img,
                tuple(box[:2].astype(int)),
                tuple(box[2:].astype(int)),
                color,
                -1,
            )
        return cv2.addWeighted(mask_img, 0.3, image, 0.7, 0)

    def prepare_model_details(self):
        # 准备模型输入输出细节
        input_info = self.session.get_inputs()[0]
        self.input_names = [input_info.name]
        self.input_height, self.input_width = input_info.shape[2:]
        self.output_names = [output.name for output in self.session.get_outputs()]


if __name__ == "__main__":
    start_time = time.time()
    utils = OnnxUtils("sim_best.onnx", ["target"], conf_threes=0.1, iou_threes=0.5)
    input_image = cv2.imread(
        "/Users/h1code2/PycharmProjects/ultralytics/360_image/images/0a45db59-27c2-4ecd-b27a-49b553bd66e6-w600-h298-4.png")
    vals, draw_image = utils.detect_and_draw(input_image, min_score=0.88, cut=True, rim=8)
    cv2.imwrite("new_image2.png", draw_image)
    print(json.dumps(vals, default=str), time.time() - start_time)
