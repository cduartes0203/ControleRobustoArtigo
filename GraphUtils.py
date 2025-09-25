import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.express as px
import pandas as pd

def PlotSingle(y,x=None,w=6,h=4,lw=1,ms=1,mrkr=None,
               title='', xname='x', yname='y', data='Data',
               pltly=True, save=False, file_name='plot.png'):
    if x== None: x=np.arange(0,len(y),1)
    if pltly:
        fig=make_subplots(rows=1,cols=1)
        trace = go.Scatter(x=x,y=y, name=data)
        fig.add_trace(trace, row=1, col=1)
        fig.update_layout(width = w*100, height = h*100, title = title)
        fig.update_xaxes(title_text=xname, row = 1, col = 1)
        fig.update_yaxes(title_text=yname, row = 1, col = 1)
        fig.show()
        if save: fig.write_image(file_name, scale=5)
    else:
        plt.figure(figsize=(w, h))  # Define o tamanho do gráfico
        plt.plot(x, y, linestyle='-', linewidth = lw, marker=mrkr, markersize=ms, color='b', label=data)  # Plota os dados
        plt.xlabel(xname)  # Nome do eixo X
        plt.ylabel(yname)  # Nome do eixo Y
        plt.title(title)  # Define o título do gráfico
        plt.grid(False)  # Adiciona grade ao gráfico
        plt.legend()  # Exibe legenda
        plt.show()  # Mostra o gráfico
        if save: plt.savefig(file_name, dpi=500)

def PlotSeries(y_arrays, x_arrays=None, w=8, h=5, lw=1.5, ms=3, mrkr=None,
               title='', xname='x', yname='y', legend_labels=None,
               pltly=True, save=False, file_name='plot.png', return_fig=False):
    """
    Plota ou cria um objeto de figura com múltiplas séries de dados.
    
    Se return_fig=True, a função retorna o objeto da figura em vez de exibi-lo.
    """
    
    if x_arrays is None:
        x_arrays = [np.arange(len(y)) for y in y_arrays]
    if legend_labels is None:
        legend_labels = [f'Série {i+1}' for i in range(len(y_arrays))]

    if pltly:
        # Para Plotly, retornamos os traços (dados) e o layout (configurações) separadamente
        traces = []
        for x, y, name in zip(x_arrays, y_arrays, legend_labels):
            trace = go.Scatter(x=x, y=y, name=name, mode='lines+markers',
                               line=dict(width=lw), marker=dict(size=ms*2))
            traces.append(trace)
        
        layout = go.Layout(width=w*100, height=h*100, title=title,
                           xaxis_title=xname, yaxis_title=yname,
                           legend_title_text='Legenda')
        
        fig = go.Figure(data=traces, layout=layout)
        
        if return_fig:
            return fig # Retorna o objeto completo da figura

        fig.show()
        if save:
            try:
                fig.write_image(file_name, scale=5)
                print(f"Gráfico salvo como '{file_name}'")
            except ValueError as e:
                print(f"Erro ao salvar a imagem: {e}")
                print("Instale 'kaleido': pip install -U kaleido")
    
    else: # Matplotlib
        fig, ax = plt.subplots(figsize=(w, h))
        
        for x, y, label in zip(x_arrays, y_arrays, legend_labels):
            ax.plot(x, y, linestyle='-', linewidth=lw, marker=mrkr, markersize=ms, label=label)
            
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        
        if return_fig:
            return fig # Retorna a figura para composição

        if save:
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo como '{file_name}'")

        plt.show()
        plt.close(fig) # Fecha a figura para liberar memória

def MultiPlot(plots_data, rows, cols, pltly=True, main_title='', 
              save=False, file_name='multiplot.png', fig_size=(12, 8)):
    """
    Combina múltiplos gráficos gerados pela PlotSeries em uma única imagem.

    Args:
        plots_data (list of dict): Uma lista de dicionários, onde cada dicionário
                                   contém os argumentos para uma chamada da PlotSeries.
        rows (int): Número de linhas na grade de subplots.
        cols (int): Número de colunas na grade de subplots.
        pltly (bool, optional): Define se usará Plotly ou Matplotlib. Defaults to True.
        main_title (str, optional): Título principal para o conjunto de gráficos.
        save (bool, optional): Se True, salva a imagem final. Defaults to False.
        file_name (str, optional): Nome do arquivo para salvar.
        fig_size (tuple, optional): Tamanho total da figura (para Matplotlib).
    """
    if len(plots_data) > rows * cols:
        print(f"Aviso: Você tem {len(plots_data)} gráficos para plotar, mas a grade é de {rows}x{cols}. Alguns gráficos não serão exibidos.")

    if pltly:
        # Pega os títulos dos subplots dos dados, se existirem
        subplot_titles = [p.get('title', '') for p in plots_data]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        for i, plot_args in enumerate(plots_data):
            if i >= rows * cols: break
            
            # Posição na grade
            row = i // cols + 1
            col = i % cols + 1
            
            # Gera a figura temporária para extrair os dados (traços)
            temp_fig = PlotSeries(pltly=True, return_fig=True, **plot_args)
            
            # Adiciona os traços da figura temporária ao subplot correto
            for trace in temp_fig.data:
                fig.add_trace(trace, row=row, col=col)
            
            # Atualiza os eixos do subplot
            fig.update_xaxes(title_text=plot_args.get('xname', 'x'), row=row, col=col)
            fig.update_yaxes(title_text=plot_args.get('yname', 'y'), row=row, col=col)

        fig.update_layout(title_text=main_title, height=fig_size[1]*100, width=fig_size[0]*100)
        fig.show()

        if save:
            fig.write_image(file_name, scale=3)
            print(f"Multi-plot salvo como '{file_name}'")

    else: # Matplotlib
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        # Garante que 'axes' seja sempre um array iterável
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= len(plots_data):
                ax.axis('off') # Esconde eixos de subplots não utilizados
                continue

            plot_args = plots_data[i]
            
            # Desempacota os argumentos para o plot
            x_arrays = plot_args.get('x_arrays')
            y_arrays = plot_args.get('y_arrays')
            legend_labels = plot_args.get('legend_labels')
            
            # Lógica de criação de dados padrão (caso não sejam fornecidos)
            if y_arrays is None: continue
            if x_arrays is None:
                x_arrays = [np.arange(len(y)) for y in y_arrays]
            if legend_labels is None:
                legend_labels = [f'Série {j+1}' for j in range(len(y_arrays))]

            # Plota os dados no eixo (ax) correto
            for x, y, label in zip(x_arrays, y_arrays, legend_labels):
                ax.plot(x, y, label=label)
            
            ax.set_title(plot_args.get('title', f'Gráfico {i+1}'))
            ax.set_xlabel(plot_args.get('xname', 'x'))
            ax.set_ylabel(plot_args.get('yname', 'y'))
            ax.grid(True)
            ax.legend()
            
        fig.suptitle(main_title, fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta para o título principal caber

        if save:
            plt.savefig(file_name, dpi=300)
            print(f"Multi-plot salvo como '{file_name}'")

        plt.show()