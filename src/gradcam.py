import torch
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image, ImageDraw, ImageFont


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output

        if output.requires_grad:
            output.register_hook(
                self._save_gradient
            )

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def generate(
        self,
        logits,
        class_index,
        original_size
    ):
        self.model.zero_grad(
            set_to_none=True
        )

        score = logits[0, class_index]

        score.backward(
            retain_graph=True
        )

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        # Importance weight for each feature map
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        width, height = original_size

        # Resize Grad-CAM to original image dimensions
        cam = F.interpolate(
            cam,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )

        cam = cam[0, 0]

        # Normalize between 0 and 1
        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:
            cam /= max_value

        return cam.cpu().numpy()


def find_defect_contour(heatmap):
    """
    Extract an approximate contour following the strongest
    local Grad-CAM activation.

    This is model interpretability, not true segmentation.
    """

    height, width = heatmap.shape

    # --------------------------------------------------
    # 1. Find strongest activation
    # --------------------------------------------------

    peak_y, peak_x = np.unravel_index(
        np.argmax(heatmap),
        heatmap.shape
    )

    # --------------------------------------------------
    # 2. Restrict analysis to area around strongest point
    # --------------------------------------------------

    window_width = max(
        50,
        int(width * 0.30)
    )

    window_height = max(
        50,
        int(height * 0.30)
    )

    x1 = max(
        0,
        peak_x - window_width // 2
    )

    x2 = min(
        width,
        peak_x + window_width // 2
    )

    y1 = max(
        0,
        peak_y - window_height // 2
    )

    y2 = min(
        height,
        peak_y + window_height // 2
    )

    local_heatmap = heatmap[
        y1:y2,
        x1:x2
    ]

    if local_heatmap.size == 0:
        return None

    # --------------------------------------------------
    # 3. Keep strongest activation region
    # --------------------------------------------------

    percentile_threshold = np.percentile(
        local_heatmap,
        88
    )

    relative_threshold = (
        local_heatmap.max() * 0.65
    )

    threshold = max(
        float(percentile_threshold),
        float(relative_threshold)
    )

    mask = (
        local_heatmap >= threshold
    ).astype(np.uint8)

    # --------------------------------------------------
    # 4. Clean small noisy regions
    # --------------------------------------------------

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------
    # 5. Find contours
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    local_peak = (
        float(peak_x - x1),
        float(peak_y - y1)
    )

    selected_contour = None

    # Prefer contour containing strongest activation
    for contour in contours:

        result = cv2.pointPolygonTest(
            contour,
            local_peak,
            False
        )

        if result >= 0:
            selected_contour = contour
            break

    # Otherwise use largest contour
    if selected_contour is None:
        selected_contour = max(
            contours,
            key=cv2.contourArea
        )

    # Ignore extremely tiny areas
    if cv2.contourArea(
        selected_contour
    ) < 10:
        return None

    # --------------------------------------------------
    # 6. Smooth the contour without converting to rectangle
    # --------------------------------------------------

    perimeter = cv2.arcLength(
        selected_contour,
        True
    )

    epsilon = (
        0.015 * perimeter
    )

    smoothed_contour = cv2.approxPolyDP(
        selected_contour,
        epsilon,
        True
    )

    # Convert from local coordinates to full image
    smoothed_contour = (
        smoothed_contour.copy()
    )

    smoothed_contour[:, :, 0] += x1
    smoothed_contour[:, :, 1] += y1

    return smoothed_contour


def closest_contour_point(
    contour_points,
    target_x,
    target_y
):
    """
    Return contour point closest to the label.
    """

    distances = []

    for x, y in contour_points:

        distance = (
            (x - target_x) ** 2
            +
            (y - target_y) ** 2
        )

        distances.append(
            distance
        )

    index = int(
        np.argmin(distances)
    )

    return contour_points[
        index
    ]


def annotate_defect_region(
    image,
    heatmap,
    defect_name
):
    """
    Draw an approximate contour around the strongest Grad-CAM
    region and place a label well above it.

    The contour follows the heatmap shape rather than using
    a rectangular bounding box.
    """

    base_image = image.convert(
        'RGBA'
    )

    image_width, image_height = (
        base_image.size
    )

    contour = find_defect_contour(
        heatmap
    )

    if contour is None:
        return base_image.convert(
            'RGB'
        )

    contour_points = [
        (
            int(point[0][0]),
            int(point[0][1])
        )
        for point in contour
    ]

    if len(contour_points) < 3:
        return base_image.convert(
            'RGB'
        )

    # --------------------------------------------------
    # SEMI-TRANSPARENT REGION
    # --------------------------------------------------

    overlay = Image.new(
        'RGBA',
        base_image.size,
        (
            0,
            0,
            0,
            0
        )
    )

    overlay_draw = ImageDraw.Draw(
        overlay
    )

    # Light transparent red fill
    overlay_draw.polygon(
        contour_points,
        fill=(
            255,
            0,
            0,
            45
        )
    )

    # Strong red outline
    overlay_draw.line(
        contour_points
        +
        [
            contour_points[0]
        ],
        fill=(
            255,
            0,
            0,
            255
        ),
        width=5
    )

    annotated = Image.alpha_composite(
        base_image,
        overlay
    ).convert(
        'RGB'
    )

    draw = ImageDraw.Draw(
        annotated
    )

    # --------------------------------------------------
    # CONTOUR BOUNDS
    # --------------------------------------------------

    xs = [
        point[0]
        for point
        in contour_points
    ]

    ys = [
        point[1]
        for point
        in contour_points
    ]

    region_min_x = min(xs)
    region_max_x = max(xs)

    region_min_y = min(ys)
    region_max_y = max(ys)

    region_center_x = int(
        np.mean(xs)
    )

    # --------------------------------------------------
    # LABEL FONT
    # --------------------------------------------------

    font_size = max(
        28,
        int(
            min(
                image_width,
                image_height
            ) * 0.048
        )
    )

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            font_size
        )

    except OSError:
        try:
            font = ImageFont.truetype(
                "DejaVuSans.ttf",
                font_size
            )
        except OSError:
            font = ImageFont.load_default()

    # Real text bounding box
    text_bbox = draw.textbbox(
        (0, 0),
        defect_name,
        font=font
    )

    text_left_offset = text_bbox[0]
    text_top_offset = text_bbox[1]
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # More padding so the text does not touch the border
    padding_x = 16
    padding_y = 12

    box_width = text_width + (2 * padding_x)
    box_height = text_height + (2 * padding_y)

    # --------------------------------------------------
    # LABEL POSITION
    # --------------------------------------------------

    edge_margin = 25

    # Keep the label well above the defect
    label_gap = 110

    label_x = region_center_x - (box_width // 2)
    label_y = region_min_y - box_height - label_gap

    # Keep inside image
    label_x = max(
        edge_margin,
        min(
            label_x,
            image_width - box_width - edge_margin
        )
    )

    label_y = max(
        edge_margin,
        label_y
    )

    label_left = label_x
    label_top = label_y
    label_right = label_x + box_width
    label_bottom = label_y + box_height

    # --------------------------------------------------
    # DRAW LABEL
    # --------------------------------------------------

    draw.rounded_rectangle(
        [
            (label_left, label_top),
            (label_right, label_bottom)
        ],
        radius=12,
        fill='white',
        outline='black',
        width=2
    )

    # Important:
    # compensate for bbox offsets so letters are not clipped
    text_x = label_x + padding_x - text_left_offset
    text_y = label_y + padding_y - text_top_offset

    draw.text(
        (text_x, text_y),
        defect_name,
        fill='black',
        font=font
    )

    # --------------------------------------------------
    # ARROW
    # --------------------------------------------------

    arrow_start = (
        (
            label_left
            +
            label_right
        ) // 2,

        label_bottom
    )

    arrow_end = closest_contour_point(
        contour_points,
        arrow_start[0],
        arrow_start[1]
    )

    # Arrow line
    draw.line(
        [
            arrow_start,
            arrow_end
        ],
        fill='black',
        width=4
    )

    # Arrow head
    dx = (
        arrow_end[0]
        -
        arrow_start[0]
    )

    dy = (
        arrow_end[1]
        -
        arrow_start[1]
    )

    length = max(
        1.0,
        (
            dx ** 2
            +
            dy ** 2
        ) ** 0.5
    )

    ux = dx / length
    uy = dy / length

    perpendicular_x = -uy
    perpendicular_y = ux

    arrow_size = 14

    arrow_left = (
        int(
            arrow_end[0]
            -
            ux * arrow_size
            +
            perpendicular_x
            *
            arrow_size
            *
            0.55
        ),

        int(
            arrow_end[1]
            -
            uy * arrow_size
            +
            perpendicular_y
            *
            arrow_size
            *
            0.55
        )
    )

    arrow_right = (
        int(
            arrow_end[0]
            -
            ux * arrow_size
            -
            perpendicular_x
            *
            arrow_size
            *
            0.55
        ),

        int(
            arrow_end[1]
            -
            uy * arrow_size
            -
            perpendicular_y
            *
            arrow_size
            *
            0.55
        )
    )

    draw.polygon(
        [
            arrow_end,
            arrow_left,
            arrow_right
        ],
        fill='black'
    )

    return annotated