"""
TopoGuard Live Monitoring Dashboard (Streamlit)
------------------------------------------
圖形化即時資料監控、分析結果視覺化與來源控制。

使用方式：
    streamlit run run_simulation_gui.py
"""

import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面配置
st.set_page_config(
    page_title="TopoGuard 即時監控",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize per-session state in one place.
for key, value in {
    "tick_count": 0,
    "anomaly_count": 0,
    "normal_count": 0,
    "history": [],
    "engine": None,
    "notifier": None,
    "running": False,
    "data_source": None,
    "source_config": None,
    "latest_result": None,
    "use_polars": True,
    "use_hybrid": True,
    "use_z3": True,
    "window_size": 300,
    "source_type": "system",
    "input_path": "",
    "batch_size": 500,
    "interval": 0.5,
    "notify_enabled": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def init_engine():
    """初始化引擎。"""
    try:
        from defensive import TopoGuardEngine

        engine = TopoGuardEngine(
            use_ebpf=False,
            use_polars=st.session_state.use_polars,
            use_hybrid=st.session_state.use_hybrid,
            use_z3=st.session_state.use_z3,
            window_size=st.session_state.window_size,
            stride=max(1, st.session_state.window_size // 2)
        )
        return engine
    except Exception as e:
        st.error(f"引擎初始化失敗: {e}")
        return None


def init_data_source():
    """Create the selected real source; input paths are mandatory for files."""
    from data_sources import create_data_source

    return create_data_source(
        st.session_state.source_type,
        input_path=st.session_state.input_path or None,
        window_size=st.session_state.window_size,
        batch_size=st.session_state.batch_size,
        poll_interval=st.session_state.interval,
    )


def run_tick():
    """執行單次 tick。"""
    if st.session_state.engine is None:
        st.session_state.engine = init_engine()

    if st.session_state.engine is None:
        return None

    st.session_state.tick_count += 1

    if st.session_state.data_source is None:
        st.error("請先選擇並啟動真實資料來源")
        st.session_state.running = False
        return None

    stream = st.session_state.data_source.collect()
    minimum = getattr(st.session_state.data_source, "min_samples", 1)
    if len(stream) < minimum:
        result = {
            "status": "WARMING_UP" if stream else "NO_DATA",
            "is_anomaly": False,
            "data_points": len(stream),
            "message": f"Collecting samples ({len(stream)}/{minimum})" if stream else "No new source data",
        }
    else:
        result = st.session_state.engine.inspect_tick(
            stream,
            replace_stream=getattr(st.session_state.data_source, "snapshot", False),
        )

    st.session_state.latest_result = result
    if (
        st.session_state.notifier is not None
        and st.session_state.notifier.send_alert(
            result,
            st.session_state.data_source.name,
            st.session_state.tick_count,
        )
    ):
        st.success("異常摘要已傳送至 webhook")
    # 統計
    if result.get("status") == "SUCCESS" and result.get('is_anomaly', False):
        st.session_state.anomaly_count += 1
    elif result.get("status") == "SUCCESS":
        st.session_state.normal_count += 1

    if result.get("status") == "SUCCESS":
        history_entry = {
            'tick': st.session_state.tick_count,
            'anomaly_score': result.get('anomaly_score', 0.0),
            'tda_score': result.get('context', {}).get('tda_score', 0.0),
            'iforest_score': result.get('context', {}).get('iforest_score', 0.0),
            'iforest_status': result.get('context', {}).get('iforest_details', {}).get('status', 'disabled'),
            'is_anomaly': result.get('is_anomaly', False),
            'severity': result.get('context', {}).get('severity', 'N/A'),
            'data_points': result.get('data_points', 0),
            'source': st.session_state.data_source.name,
        }
        st.session_state.history.append(history_entry)
        if len(st.session_state.history) > 100:
            st.session_state.history = st.session_state.history[-100:]

    return result


def create_anomaly_score_chart():
    """創建異常分數趨勢圖。"""
    if not st.session_state.history:
        return None

    df = pd.DataFrame(st.session_state.history)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['tick'],
        y=df['anomaly_score'],
        mode='lines+markers',
        name='異常分數',
        line=dict(color='blue', width=2)
    ))

    # 標記異常點
    anomaly_df = df[df['is_anomaly']]
    if not anomaly_df.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_df['tick'],
            y=anomaly_df['anomaly_score'],
            mode='markers',
            name='檢測到異常',
            marker=dict(color='red', size=10, symbol='x')
        ))

    fig.update_layout(
        title='異常分數趨勢',
        xaxis_title='Tick',
        yaxis_title='異常分數',
        hovermode='x unified'
    )

    return fig


def create_statistics_chart():
    """創建統計圖表。"""
    if st.session_state.tick_count == 0:
        return None

    labels = ['正常', '異常']
    values = [st.session_state.normal_count, st.session_state.anomaly_count]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker_colors=['green', 'red']
    )])

    fig.update_layout(title='檢測統計')

    return fig


def main():
    """主函數。"""
    st.title("TopoGuard 即時監控")
    st.markdown("---")

    # 側邊欄配置
    with st.sidebar:
        st.header("⚙️ 配置")

        # 初始化配置
        st.session_state.use_polars = st.checkbox("啟用 Polars", value=st.session_state.use_polars)
        st.session_state.use_hybrid = st.checkbox("啟用混合 AI", value=st.session_state.use_hybrid)
        st.session_state.use_z3 = st.checkbox("啟用 Z3 驗證", value=st.session_state.use_z3)
        st.session_state.notify_enabled = st.checkbox(
            "啟用 Slack/Discord webhook 通知",
            value=st.session_state.notify_enabled,
        )
        if st.session_state.notify_enabled and st.session_state.notifier is None:
            try:
                from notifications import WebhookNotifier

                st.session_state.notifier = WebhookNotifier.from_environment()
            except ValueError as exc:
                st.error(f"Webhook 設定錯誤：{exc}")
                st.session_state.running = False
        elif not st.session_state.notify_enabled:
            st.session_state.notifier = None
        st.session_state.window_size = st.slider("視窗大小", 10, 500, st.session_state.window_size)
        st.session_state.source_type = st.selectbox(
            "即時資料來源",
            ["system", "ebpf", "prometheus", "web-log", "csv", "json", "jsonl", "zeek", "pcap"],
            index=["system", "ebpf", "prometheus", "web-log", "csv", "json", "jsonl", "zeek", "pcap"].index(st.session_state.source_type),
        )
        file_sources = {"web-log", "csv", "json", "jsonl", "zeek", "pcap"}
        needs_input = st.session_state.source_type in file_sources | {"prometheus"}
        if needs_input:
            input_label = (
                "Prometheus /metrics endpoint URL"
                if st.session_state.source_type == "prometheus"
                else "輸入檔案路徑"
            )
            st.session_state.input_path = st.text_input(input_label, st.session_state.input_path)
        if st.session_state.source_type in file_sources:
            st.session_state.batch_size = st.number_input(
                "每批讀取筆數", min_value=1, max_value=5000, value=st.session_state.batch_size
            )
        st.session_state.interval = st.number_input(
            "採集間隔（秒）", min_value=0.0, max_value=60.0, value=st.session_state.interval
        )

        source_config = (
            st.session_state.source_type,
            st.session_state.input_path if needs_input else "",
            st.session_state.window_size,
            st.session_state.batch_size,
            st.session_state.interval,
        )
        if st.session_state.source_config != source_config:
            if st.session_state.data_source is not None:
                st.session_state.data_source.stop()
            st.session_state.data_source = None
            st.session_state.source_config = source_config
            st.session_state.running = False

        st.markdown("---")

        # 控制按鈕
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ 開始/繼續"):
                try:
                    if st.session_state.notify_enabled and st.session_state.notifier is None:
                        raise RuntimeError("請先設定 TOPOGUARD_WEBHOOK_URL")
                    if st.session_state.data_source is None:
                        st.session_state.data_source = init_data_source()
                    if st.session_state.engine is None:
                        st.session_state.engine = init_engine()
                    st.session_state.running = st.session_state.engine is not None
                except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
                    st.error(f"資料來源初始化失敗：{exc}")
                    st.session_state.running = False
        with col2:
            if st.button("⏸️ 暫停"):
                st.session_state.running = False

        if st.button("🔄 重置"):
            st.session_state.tick_count = 0
            st.session_state.anomaly_count = 0
            st.session_state.normal_count = 0
            st.session_state.history = []
            st.session_state.latest_result = None
            if st.session_state.data_source is not None:
                st.session_state.data_source.stop()
            st.session_state.data_source = None
            st.session_state.engine = None
            st.session_state.running = False
            st.rerun()

        st.markdown("---")

        # 統計資訊
        st.header("📊 統計")
        st.metric("總 Tick 數", st.session_state.tick_count)
        st.metric("檢測到異常", st.session_state.anomaly_count)
        st.metric("正常狀態", st.session_state.normal_count)
        if st.session_state.tick_count > 0:
            anomaly_rate = st.session_state.anomaly_count / st.session_state.tick_count * 100
            st.metric("異常率", f"{anomaly_rate:.2f}%")

    # 主內容區
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 異常分數趨勢")
        chart = create_anomaly_score_chart()
        if chart:
            st.plotly_chart(chart, width="stretch", alt="異常分數隨監控 tick 的變化")
        else:
            st.info("選擇即時資料來源後，點擊「開始/繼續」")

    with col2:
        st.subheader("🥧 檢測統計")
        pie_chart = create_statistics_chart()
        if pie_chart:
            st.plotly_chart(pie_chart, width="stretch", alt="正常與異常事件比例")

    # 最新結果
    st.markdown("---")
    st.subheader("🔍 最新檢測結果")

    if st.session_state.history:
        latest = st.session_state.history[-1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tick", latest['tick'])
        col2.metric("異常分數", f"{latest['anomaly_score']:.4f}")
        col3.metric("狀態", "🚨 異常" if latest['is_anomaly'] else "✅ 正常")
        col4.metric("嚴重程度", latest['severity'])
    if st.session_state.history:
        latest_history = st.session_state.history[-1]
        st.caption(
            f"TDA={latest_history['tda_score']:.4f} · "
            f"Isolation Forest={latest_history['iforest_score']:.4f} "
            f"({latest_history['iforest_status']})"
        )
        st.caption(f"來源：{latest['source']}；資料點：{latest['data_points']}")
    elif st.session_state.latest_result:
        st.info(st.session_state.latest_result.get("message", st.session_state.latest_result.get("status", "尚無有效資料")))

    # 自動運行循環
    if st.session_state.running:
        try:
            result = run_tick()
        except (RuntimeError, ValueError) as exc:
            st.error(f"監控或通知失敗：{exc}")
            st.session_state.running = False
            result = None
        if result:
            if result.get("status") == "ERROR":
                st.error(result.get("message", "監控失敗"))
                st.session_state.running = False
            elif getattr(st.session_state.data_source, "finished", False):
                st.session_state.data_source.stop()
                st.session_state.data_source = None
                st.session_state.running = False
            else:
                time.sleep(st.session_state.interval)
            st.rerun()


if __name__ == "__main__":
    main()
