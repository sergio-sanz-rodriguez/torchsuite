import gradio as gr
import html as _html
import time
import plotly.graph_objects as go


dark_soft = gr.themes.Soft(
    primary_hue="neutral",
    secondary_hue="neutral"
).set(
    body_background_fill="#000000",         # main background
    body_text_color="#EEEEEE",              # text color
    background_fill_primary="#000000",      # blocks, panels
    background_fill_secondary="#000000",    # secondary containers
    block_background_fill="#000000",        # nested blocks
    border_color_primary="#333333",         # subtle borders
)

CSS = """
/* Make labels bigger for any component with elem_classes=["plot-lg-label"] */
.plot-lg-label .wrap > label,
.plot-lg-label label,
.plot-lg-label .label {   /* covers newer Gradio builds where label uses a .label div */
  font-size: 1.5rem !important;
  font-weight: 800 !important;
  text-align: center !important;
  color: #eee !important;
}
"""

legend_html = """
<div style="
    position: fixed;
    top: 20px;
    right: 40px;
    background: rgba(0,0,0,0.6);
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    font-family: sans-serif;
    z-index: 1000;
">
  <b>Legend</b><br>
  <span style='color: orange;'>⬤</span> Train<br>
  <span style='color: blue;'>⬤</span> Validation
</div>
"""


class EnginePlotApp:
    def __init__(self, engine, title="Training Plots",
                 interval: float = 0.5, min_cooldown: float = 0.5,
                 port: int = 7860, server_name: str = "0.0.0.0"):
        self.engine = engine
        self.title = title
        self.interval = interval
        self.min_cooldown = min_cooldown
        self.port = port
        self.server_name = server_name
        self.demo = None
        self._launch = None

    def _fetch_all(self):
        # manual refresh: return all seven figs
        figs = getattr(self.engine, "_ui_plotly_cache", None)
        if not figs or any(f is None for f in figs):
            # build once if needed
            self.engine._display_results(return_html=False, clear_notebook=False)
            figs = self.engine._ui_plotly_cache
        return figs

    def _fetch_if_new(self, last_epoch_sent: int, last_push_ts: float):
        now = time.time()
        curr_epoch = int(getattr(self.engine, "_ui_last_epoch_built", 0) or 0)
        has_new = bool(getattr(self.engine, "_ui_new_data", False))
        if has_new and curr_epoch > (last_epoch_sent or 0) and (now - (last_push_ts or 0.0)) >= self.min_cooldown:
            self.engine._ui_new_data = False
            figs = self._fetch_all()
            return (*figs, curr_epoch, now)
        # no change -> keep current plots
        return *(gr.update() for _ in range(7)), last_epoch_sent, last_push_ts
    
    def _make_legend_fig(self):
        legend_fig = go.Figure()
        legend_fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            name="Train",
            line=dict(color="orange", width=3)
        ))
        legend_fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            name="Validation",
            line=dict(color="blue", width=3)
        ))
        legend_fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                x=0, y=1,
                xanchor="left",
                yanchor="top",
                font=dict(size=13),
                bgcolor="rgba(0,0,0,0)"
            ),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=200,
        )
        return legend_fig

    def build(self):

        with gr.Blocks(
            title="Training Dashboard",
            theme=gr.themes.Default(
                primary_hue="neutral",
                secondary_hue="neutral")
            ) as demo:        
            #gr.Markdown("## 🧠 Training Plots (7 independent figures — no flicker)")
            #gr.HTML(legend_html)
    
            with gr.Row():
                loss = gr.Plot(show_label=False) #(label="Loss")
                acc  = gr.Plot(show_label=False) #(label="Accuracy")
                f1   = gr.Plot(show_label=False) #(label="F1-score")

            with gr.Row():
                fpr  = gr.Plot(show_label=False) #(label="FPR @ recall")
                pauc = gr.Plot(show_label=False) #(label="pAUC @ recall")
                time_ = gr.Plot(show_label=False) #(label="Time (s)")
                lr   = gr.Plot(show_label=False) #(label="Learning Rate")
                
            refresh = gr.Button("🔄 Refresh")

            # state for throttling
            last_epoch_sent = gr.State(value=0)
            last_push_ts = gr.State(value=0.0)

            def _force_refresh(_e, _t):
                figs = self._fetch_all()
                now = time.time()
                curr = int(getattr(self.engine, "_ui_last_epoch_built", 0) or 0)
                self.engine._ui_new_data = False
                return (*figs, curr, now)

            refresh.click(
                _force_refresh,
                inputs=[last_epoch_sent, last_push_ts],
                outputs=[loss, acc, f1, fpr, pauc, time_, lr, last_epoch_sent, last_push_ts]
            )

            gr.Timer(self.interval).tick(
                self._fetch_if_new,
                inputs=[last_epoch_sent, last_push_ts],
                outputs=[loss, acc, f1, fpr, pauc, time_, lr, last_epoch_sent, last_push_ts]
            )

        self.demo = demo
        return demo

    def start(self):
        if self.demo is None:
            self.build()
        self._launch = self.demo.launch(
            prevent_thread_lock=True,
            show_error=True,
            inbrowser=True,
            inline=False,
            quiet=True,
            #favicon_path="assets/my_logo.png"
            # server_name=self.server_name,
            # server_port=self.port,
        )
        return self

    def stop(self):
        try:
            if self._launch is not None:
                self._launch.close()
        except Exception:
            pass
