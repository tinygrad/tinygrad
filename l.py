import plotly.express as px
import pandas as pd

colors = ["#6D67E4","#46C2CB","#F2F7A1", "#98f5e1","#f1c0e8","#a3c4f3","#fbf8cc","#cfbaf0","#fde4cf","#90dbf4","#ffcfd2","#8eecf5","#b9fbc0"]
def layoutizer(fig,xaxis_title=None,yaxis_title=None,legend_title=None,is_line=False,ticksuffix=None,tickprefix=None):
    c = "white"; bg = "black"
    fig.update_layout(plot_bgcolor=bg, paper_bgcolor=bg)
    fig.update_layout(xaxis_title=xaxis_title,yaxis_title=yaxis_title,font=dict(color=c),legend_title=legend_title)
    fig.update_xaxes(showline=True, linewidth=1, linecolor=c); fig.update_yaxes(showline=True, linewidth=1, linecolor=c); fig.update_yaxes(ticksuffix=ticksuffix, tickprefix=tickprefix); fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    if is_line: fig.update_traces(line=dict(width=3))
    return fig

y = "time"; title = "Time (ms)"
df = pd.read_csv("loads.csv")
fig = px.bar(df, x="layer", color="op", y=y, title="LLaMa 2-7B Q4.0", color_discrete_sequence=colors)
layoutizer(fig, xaxis_title="Layer", yaxis_title=title, legend_title="Operation").show()
