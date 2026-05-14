from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from inference import infer_with_details, load_config, load_model, process_input_data
from utility.create_redundant_representation import recover_from_ric


APP_TITLE = "CARE-PD UPDRS Predictor"
UPLOAD_DIR = Path("tmp/frontend_uploads")
DEFAULT_CONFIG = "configs/momask/inference_config.json"
DEFAULT_MODEL = "assets/Pretrained_checkpoints/momask/best_model.pt"
HUMANML3D_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12),
    (12, 15), (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
]


def ensure_upload_dir():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner=False)
def get_model_bundle(config_path, model_path):
    config = load_config(config_path)
    model = load_model(config, model_path)
    return config, model


def save_uploaded_file(uploaded_file):
    ensure_upload_dir()
    suffix = Path(uploaded_file.name).suffix or ".npy"
    with tempfile.NamedTemporaryFile(
        dir=UPLOAD_DIR,
        prefix="care_pd_",
        suffix=suffix,
        delete=False,
    ) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def load_uploaded_sequence(file_path):
    sequence = np.load(file_path)
    if sequence.ndim == 3:
        sequence = sequence[0]
    if sequence.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D npy array, got shape {sequence.shape}")
    return sequence


def recover_sequence_3d(sequence_2d):
    if sequence_2d.shape[1] % 3 == 0 and sequence_2d.shape[1] != 263:
        return sequence_2d.reshape(sequence_2d.shape[0], sequence_2d.shape[1] // 3, 3)
    if sequence_2d.shape[1] != 263:
        raise ValueError(
            "Expected HumanML3D features with dim 263, or a flattened xyz skeleton."
        )
    seq_tensor = torch.from_numpy(sequence_2d).unsqueeze(0).float()
    sequence_3d = recover_from_ric(seq_tensor, 22).squeeze(0).cpu().numpy()
    return sequence_3d


def compute_metrics(sequence_2d, sequence_3d, fps):
    frame_count = int(sequence_2d.shape[0])
    feature_dim = int(sequence_2d.shape[1])
    joint_velocity = np.diff(sequence_3d, axis=0)
    joint_speed = np.linalg.norm(joint_velocity, axis=2).mean(axis=1) if len(sequence_3d) > 1 else np.array([0.0])

    root_xy = sequence_3d[:, 0, [0, 2]]
    root_disp = np.diff(root_xy, axis=0)
    root_speed = np.linalg.norm(root_disp, axis=1) if len(root_xy) > 1 else np.array([0.0])
    root_path_length = float(root_speed.sum())

    left_foot_speed = None
    right_foot_speed = None
    asymmetry = None
    if sequence_3d.shape[1] > 11 and len(sequence_3d) > 1:
        left_foot_speed = np.linalg.norm(np.diff(sequence_3d[:, 10, :], axis=0), axis=1)
        right_foot_speed = np.linalg.norm(np.diff(sequence_3d[:, 11, :], axis=0), axis=1)
        asymmetry = float(np.mean(np.abs(left_foot_speed - right_foot_speed)))

    return {
        "frame_count": frame_count,
        "feature_dim": feature_dim,
        "duration_sec": frame_count / fps if fps else None,
        "avg_joint_speed": float(joint_speed.mean()),
        "peak_joint_speed": float(joint_speed.max()),
        "root_path_length": root_path_length,
        "mean_root_speed": float(root_speed.mean()) if len(root_speed) else 0.0,
        "movement_energy": float(np.square(joint_speed).mean()),
        "joint_speed_series": joint_speed,
        "root_xy": root_xy,
        "root_speed_series": root_speed,
        "left_foot_speed": left_foot_speed,
        "right_foot_speed": right_foot_speed,
        "left_right_asymmetry": asymmetry,
    }


def build_probability_df(result):
    return pd.DataFrame(
        {
            "score": list(range(len(result["class_probabilities"]))),
            "probability": result["class_probabilities"],
        }
    )


def build_joint_speed_df(metrics, fps):
    series = metrics["joint_speed_series"]
    time_axis = np.arange(len(series)) / fps if fps else np.arange(len(series))
    return pd.DataFrame({"time": time_axis, "speed": series})


def build_root_path_df(metrics):
    root_xy = metrics["root_xy"]
    return pd.DataFrame({"x": root_xy[:, 0], "z": root_xy[:, 1], "frame": np.arange(len(root_xy))})


def build_foot_speed_df(metrics, fps):
    if metrics["left_foot_speed"] is None or metrics["right_foot_speed"] is None:
        return None
    time_axis = np.arange(len(metrics["left_foot_speed"])) / fps if fps else np.arange(len(metrics["left_foot_speed"]))
    return pd.DataFrame(
        {
            "time": np.concatenate([time_axis, time_axis]),
            "speed": np.concatenate([metrics["left_foot_speed"], metrics["right_foot_speed"]]),
            "side": ["left"] * len(time_axis) + ["right"] * len(time_axis),
        }
    )


def make_skeleton_figure(sequence_3d, frame_idx):
    frame = sequence_3d[frame_idx]
    x_coords = frame[:, 0]
    y_coords = frame[:, 1]  # vertical axis: body height
    z_coords = frame[:, 2]  # forward axis: walking direction
    x_lines = []
    y_lines = []
    z_lines = []
    valid_edges = [(a, b) for a, b in HUMANML3D_EDGES if a < frame.shape[0] and b < frame.shape[0]]
    for start, end in valid_edges:
        x_lines.extend([x_coords[start], x_coords[end], None])
        y_lines.extend([y_coords[start], y_coords[end], None])
        z_lines.extend([z_coords[start], z_coords[end], None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers",
            marker={"size": 5, "color": "#0f766e"},
            name="joints",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line={"width": 5, "color": "#f97316"},
            name="skeleton",
        )
    )
    fig.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "X (left-right)"},
            "yaxis": {"title": "Y (height)"},
            "zaxis": {"title": "Z (forward)"},
            "camera": {
                "up": {"x": 0, "y": 1, "z": 0},
                "eye": {"x": 1.5, "y": 1.0, "z": 1.5},
            },
        },
    )
    return fig


def make_summary_table(metrics):
    rows = [
        ("Frames", metrics["frame_count"]),
        ("Feature dim", metrics["feature_dim"]),
        ("Duration (sec)", None if metrics["duration_sec"] is None else round(metrics["duration_sec"], 2)),
        ("Avg joint speed", round(metrics["avg_joint_speed"], 4)),
        ("Peak joint speed", round(metrics["peak_joint_speed"], 4)),
        ("Root path length", round(metrics["root_path_length"], 4)),
        ("Movement energy", round(metrics["movement_energy"], 4)),
    ]
    if metrics["left_right_asymmetry"] is not None:
        rows.append(("Left-right asymmetry", round(metrics["left_right_asymmetry"], 4)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def render_analysis_insights(result, metrics):
    top_score = result["predicted_updrs_score"]
    top_prob = max(result["class_probabilities"])

    st.markdown("### Step 3. Prediction Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted UPDRS", top_score)
    col2.metric("Top-class confidence", f"{top_prob:.1%}")
    col3.metric("Frames used", metrics["frame_count"])

    insight_rows = [
        {
            "dimension": "Model output",
            "note": f"Current sample is most likely in score class {top_score}.",
        },
        {
            "dimension": "Temporal dynamics",
            "note": f"Average joint speed is {metrics['avg_joint_speed']:.4f}, with a peak of {metrics['peak_joint_speed']:.4f}.",
        },
        {
            "dimension": "Spatial progression",
            "note": f"Root trajectory path length is {metrics['root_path_length']:.4f}.",
        },
    ]
    if metrics["left_right_asymmetry"] is not None:
        insight_rows.append(
            {
                "dimension": "Symmetry proxy",
                "note": f"Left-right foot asymmetry proxy is {metrics['left_right_asymmetry']:.4f}.",
            }
        )
    st.dataframe(pd.DataFrame(insight_rows), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🦴", layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload a HumanML3D-format .npy skeleton and inspect UPDRS prediction, probabilities, and gait analysis charts.")

    with st.sidebar:
        st.subheader("Model")
        config_path = st.text_input("Config path", value=DEFAULT_CONFIG)
        model_path = st.text_input("Checkpoint path", value=DEFAULT_MODEL)
        fps = st.number_input("Playback / analysis FPS", min_value=1, max_value=120, value=30, step=1)

        st.subheader("Future Video Hook")
        st.text_area(
            "Video-to-skeleton command template",
            value="python your_video2skeleton.py --input {video_path} --output {npy_path}",
            help="Reserved for your existing video-to-skeleton script. This prototype focuses on direct .npy upload first.",
            height=100,
        )

    st.markdown("## Step 1. Upload Skeleton Input")
    uploaded_file = st.file_uploader("Upload a HumanML3D .npy file", type=["npy"])
    if uploaded_file is None:
        st.info("Upload a .npy file to begin.")
        return

    saved_path = save_uploaded_file(uploaded_file)
    st.success(f"Saved upload to {saved_path}")

    try:
        sequence_2d = load_uploaded_sequence(saved_path)
        sequence_3d = recover_sequence_3d(sequence_2d)
        metrics = compute_metrics(sequence_2d, sequence_3d, fps)
    except Exception as exc:
        st.error(f"Failed to parse the uploaded skeleton: {exc}")
        return

    st.markdown("## Step 2. Skeleton Preview and Motion Features")
    left_col, right_col = st.columns([1, 1.4])
    with left_col:
        st.dataframe(make_summary_table(metrics), use_container_width=True, hide_index=True)
        frame_idx = st.slider("Preview frame", 0, len(sequence_3d) - 1, min(10, len(sequence_3d) - 1))
        st.plotly_chart(make_skeleton_figure(sequence_3d, frame_idx), use_container_width=True)

    with right_col:
        prob_placeholder = st.empty()
        speed_df = build_joint_speed_df(metrics, fps)
        root_df = build_root_path_df(metrics)
        st.plotly_chart(
            px.line(
                speed_df,
                x="time",
                y="speed",
                title="Average Joint Speed Over Time",
                template="plotly_white",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(
                root_df,
                x="x",
                y="z",
                markers=True,
                title="Root Trajectory on the Ground Plane",
                template="plotly_white",
            ),
            use_container_width=True,
        )
        foot_df = build_foot_speed_df(metrics, fps)
        if foot_df is not None:
            st.plotly_chart(
                px.line(
                    foot_df,
                    x="time",
                    y="speed",
                    color="side",
                    title="Left vs Right Foot Speed Proxy",
                    template="plotly_white",
                ),
                use_container_width=True,
            )

    st.markdown("## Step 3. Run UPDRS Inference")
    if not Path(config_path).exists():
        st.error(f"Config not found: {config_path}")
        return
    if not Path(model_path).exists():
        st.error(f"Checkpoint not found: {model_path}")
        return

    if st.button("Predict UPDRS", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading model and running inference..."):
                config, model = get_model_bundle(config_path, model_path)
                pose_data, metadata, valid_mask = process_input_data(str(saved_path), config)
                result = infer_with_details(model, pose_data, metadata, valid_mask, config)
                result.update(
                    {
                        "input_file": str(saved_path),
                        "input_filename": uploaded_file.name,
                        "model_path": model_path,
                        "config_path": config_path,
                        "dataset": config.get("dataset"),
                        "backbone": config.get("backbone", "momask"),
                    }
                )
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

        prob_df = build_probability_df(result)
        prob_placeholder.plotly_chart(
            px.bar(
                prob_df,
                x="score",
                y="probability",
                text_auto=".2f",
                title="Class Probability Distribution",
                template="plotly_white",
            ),
            use_container_width=True,
        )
        render_analysis_insights(result, metrics)

        st.markdown("### Prediction JSON")
        st.json(result)


if __name__ == "__main__":
    main()
