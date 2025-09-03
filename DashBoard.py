import random
import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html, Input, Output, State
from figure_generator import mds_handler, correlation_handler, pca_handler, RawData, area_plot

raw_data = RawData.raw_data
number_data = RawData.number_data

# 初始化dash
app = dash.Dash(__name__)

# 定义布局
app.layout = html.Div([
    html.Div([
        # 直方图
        html.Div([
            dcc.Dropdown(
                id='variable-dropdown',
                options=[
                    {'label': col, 'value': col} for col in number_data.columns
                ],
                value='HP'  # 默认选择的变量
            ),
            dcc.Graph(id='histogram'),
            dcc.Store(id='highlight-color', data=None),
            dcc.Store(id='selected-bars', data=[])  # 存储已选择的柱子和颜色
        ], style={'width': '50%', 'display': 'inline-block', 'height': '50vh'}),  # 使用视窗高度的50%

        # 第一张图像
        html.Div([
            dcc.Graph(
                id='mds-plot',
                figure=mds_handler.plot_data_by_euclidian(None, None, None)
            ),
        ], style={'display': 'inline-block'}),

    ], style={'width': '100%', 'display': 'flex'}),

    html.Div([
        # 第二张图像
        html.Div([
            dcc.Graph(
                id='parallel',
                figure=correlation_handler.plot_parallel_axes(None, None, None)
            )
        ], style={'width': '35%', 'display': 'inline-block'}),

        # 第三张图像
        html.Div([
            dcc.Graph(
                id='pca-biplot',
                figure=pca_handler.plot_biplot(None, None, None)  # Pass None to get the figure object
            ),

        ], style={'width': '30%','display': 'inline-block'}),

        # 第四张图像
        html.Div([
            dcc.Graph(
                id='area-plot',
                figure=area_plot(),
            ),
        ], style={'display': 'inline-block'}),

    ], style={'width': '100%', 'display': 'flex'}),

])
#回调函数
@app.callback(
    [Output('histogram', 'figure'),
     Output('selected-bars', 'data'),
     Output('mds-plot', 'figure'),
     Output('parallel', 'figure'),
     Output('pca-biplot', 'figure'),
     Output('highlight-color', 'data')
     ],
    [Input('variable-dropdown', 'value'),
     Input('histogram', 'clickData')],
    [State('selected-bars', 'data')]
)

def update_charts(selected_variable, click_data, selected_bars_data):

    highlight_color = None
    data_in_range=None

    ctx = dash.callback_context

    # 如果变量选择改变，则重置选中的颜色
    if ctx.triggered[0]['prop_id'] == 'variable-dropdown.value':
        selected_bars_data = []

    if selected_variable is None:
        empty_fig = go.Figure()
        return empty_fig, selected_bars_data, empty_fig, empty_fig, empty_fig, None

    # 计算分箱
    data_min = number_data[selected_variable].min()
    data_max = number_data[selected_variable].max()
    bin_size = (data_max - data_min) / 20  # 设置为有20个bins
    bins = [(data_min + bin_size * i, data_min + bin_size * (i + 1)) for i in range(20)]

    # 生成随机颜色的函数
    def generate_random_color():
        return f"rgba({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)}, 0.6)"

    # 如果有点击事件，更新选中的直方图块的数据
    if click_data and not ctx.triggered[0]['prop_id'] == 'variable-dropdown.value':
        x_value = click_data['points'][0]['x']
        # 找到被点击的分箱索引
        selected_index = next((i for i, b in enumerate(bins) if b[0] <= x_value < b[1]), None)
        # 如果分箱还没有被选中，为其分配一个颜色
        if selected_index is not None and all(selected_index != d['index'] for d in selected_bars_data):
            selected_bars_data.append({'index': selected_index, 'color': generate_random_color()})
            bin_range = bins[selected_index]
            # 过滤出原始数据中落在这个范围内的数据
            data_in_range = number_data[(number_data[selected_variable] >= bin_range[0]) &
                                        (number_data[selected_variable] < bin_range[1])]
            data_in_range.to_csv('data.csv')
            highlight_color = selected_bars_data[-1]['color'] if selected_bars_data else None

        # print(data_in_range)

    # 设置柱子的颜色
    colors = ['lightgrey'] * 20  # 初始设置所有颜色为灰色
    for d in selected_bars_data:
        # 如果某个bin已经被选中，则设置它的颜色
        colors[d['index']] = d['color']


    if selected_bars_data:
        # 计算需要highlight的数据点索引
        selected_indices = [int(index) for bar in selected_bars_data for index in range(*map(int, bins[bar['index']]))]
        highlight_color = highlight_color  # 可以根据需要设置颜色
    else:
        selected_indices = []
        highlight_color = None

    # 更新其他图表
    mds_figure = mds_handler.plot_data_by_euclidian(None, data_in_range, highlight_color)
    parallel_figure = correlation_handler.plot_parallel_axes(None, data_in_range, highlight_color)
    pca_biplot_figure = pca_handler.plot_biplot(None, data_in_range, highlight_color)
    # 创建直方图
    fig = go.Figure(data=[go.Histogram(
        x=number_data[selected_variable],
        xbins=dict(start=data_min, end=data_max, size=bin_size),
        marker_color=colors  # 设置颜色
    )])

    # 更新布局
    fig.update_layout(
        title=f'Histogram of {selected_variable}',
        xaxis_title=selected_variable,
        yaxis_title='Count',
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.2
    )

    return fig, selected_bars_data, mds_figure, parallel_figure, pca_biplot_figure, highlight_color

# 运行
if __name__ == '__main__':
    app.run_server(debug=True)
