import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO


# ---------------- YOLOv8 INFERENCE ----------------
def yolov8_inference(image, video, model_id, image_size, conf_threshold):
    model = YOLO(f"{model_id}.pt")

    # ---------- IMAGE ----------
    if image is not None:
        results = model.predict(
            source=image,
            imgsz=image_size,
            conf=conf_threshold
        )
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None

    # ---------- VIDEO ----------
    else:
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                imgsz=image_size,
                conf=conf_threshold
            )
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_path


def yolov8_examples(image, model_id, image_size, conf_threshold):
    annotated_image, _ = yolov8_inference(
        image, None, model_id, image_size, conf_threshold
    )
    return annotated_image


# ---------------- GRADIO APP ----------------
def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)

                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type"
                )

                model_id = gr.Dropdown(
                    label="YOLOv8 Model",
                    choices=[
                        "yolov8n",
                        "yolov8s",
                        "yolov8m",
                        "yolov8l",
                        "yolov8x"
                    ],
                    value="yolov8m"
                )

                image_size = gr.Slider(
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                    label="Image Size"
                )

                conf_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                    label="Confidence Threshold"
                )

                detect_btn = gr.Button("Detect Objects")

            with gr.Column():
                output_image = gr.Image(label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        # ---------- TOGGLE INPUT ----------
        def update_visibility(input_type):
            return (
                gr.update(visible=input_type == "Image"),
                gr.update(visible=input_type == "Video"),
                gr.update(visible=input_type == "Image"),
                gr.update(visible=input_type == "Video"),
            )

        input_type.change(
            fn=update_visibility,
            inputs=input_type,
            outputs=[image, video, output_image, output_video]
        )

        # ---------- RUN ----------
        def run(image, video, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov8_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolov8_inference(None, video, model_id, image_size, conf_threshold)

        detect_btn.click(
            fn=run,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video]
        )

        # ---------- EXAMPLES ----------
        gr.Examples(
            examples=[
                ["ultralytics/assets/bus.jpg", "yolov8s", 640, 0.25],
                ["ultralytics/assets/zidane.jpg", "yolov8s", 640, 0.25],
            ],
            fn=yolov8_examples,
            inputs=[image, model_id, image_size, conf_threshold],
            outputs=output_image,
            cache_examples=False
        )


# ---------------- MAIN ----------------
with gr.Blocks() as gradio_app:
    gr.HTML("<h1 style='text-align:center'>YOLOv8 Object Detection</h1>")
    gr.HTML(
        "<h3 style='text-align:center'>"
        "<a href='https://github.com/ultralytics/ultralytics' target='_blank'>GitHub</a>"
        "</h3>"
    )
    app()

if __name__ == "__main__":
    gradio_app.launch()