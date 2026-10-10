import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from DataManager import DataManager
from KMeansClusterer import KMeansClusterer
from MaximinClusterer import MaximinClusterer


BG = '#f5f7fa'
CARD = '#ffffff'
TEXT = '#1f2937'
MUTED = '#6b7280'
ACCENT = '#2563eb'
GREEN = '#10b981'
RED = '#ef4444'
PURPLE = '#8b5cf6'

CLUSTER_COLORS = ['#ef4444', '#2563eb', '#10b981',
                  '#f59e0b', '#8b5cf6', '#06b6d4',
                  '#ec4899', '#64748b']

ECONOMY_MAX_PRICE = 12.0
PREMIUM_MIN_PRICE = 60.0
ECONOMY_MIN_SALES = 500.0


def get_cluster_name(centroid):
    price = centroid[0]
    sales = centroid[1]
    if price < ECONOMY_MAX_PRICE and sales > ECONOMY_MIN_SALES:
        return 'Эконом-сегмент'
    elif price > PREMIUM_MIN_PRICE:
        return 'Премиум-сегмент'
    else:
        return 'Средний сегмент'


def reorder_clusters(labels, centroids):
    order = np.argsort(centroids[:, 0])
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    new_labels = np.array([mapping[int(l)] for l in labels])
    new_centroids = centroids[order]
    return new_labels, new_centroids


class ClusterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Кластеризация товаров")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 750)
        self.root.configure(bg=BG)

        self.dm = DataManager()
        self.kmeans = KMeansClusterer(n_clusters=3, random_state=42)
        self.maximin = MaximinClusterer(threshold=1.2, max_clusters=3)

        self.build_ui()
        self.refresh_table()
        self.draw_initial()

    def build_ui(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill=tk.X, padx=25, pady=(18, 10))

        tk.Label(header, text="Кластеризация товаров магазина электроники",
                 font=('Segoe UI', 17, 'bold'),
                 bg=BG, fg=TEXT).pack(anchor='w')
        tk.Label(header, text="Признаки: цена (тыс. руб.)  ×  количество продаж в месяц",
                 font=('Segoe UI', 10),
                 bg=BG, fg=MUTED).pack(anchor='w', pady=(2, 0))

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 20))

        left = tk.Frame(body, bg=BG, width=430)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))
        left.pack_propagate(False)

        self.build_scrollable_left(left)

        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.build_plots(right)
        self.build_info(right)

    def build_scrollable_left(self, parent):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win, width=e.width))

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def wheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")
        canvas.bind_all("<MouseWheel>", wheel)

        self.build_table(inner)
        self.build_test_form(inner)
        self.build_actions(inner)

    def card(self, parent, title=None, subtitle=None):
        wrapper = tk.Frame(parent, bg=CARD)
        wrapper.pack(fill=tk.X, pady=(0, 12))
        if title:
            tk.Label(wrapper, text=title, font=('Segoe UI', 11, 'bold'),
                     bg=CARD, fg=TEXT).pack(anchor='w', padx=18, pady=(14, 2))
        if subtitle:
            tk.Label(wrapper, text=subtitle, font=('Segoe UI', 9),
                     bg=CARD, fg=MUTED, wraplength=380, justify='left')\
                .pack(anchor='w', padx=18, pady=(0, 6))
        inner = tk.Frame(wrapper, bg=CARD)
        inner.pack(fill=tk.X, padx=18, pady=(4, 14))
        return wrapper, inner

    def build_table(self, parent):
        wrapper, _ = self.card(
            parent,
            "Обучающая выборка",
            "33 товара, на которых алгоритм ищет кластеры."
        )

        tf = tk.Frame(wrapper, bg=CARD)
        tf.pack(fill=tk.X, padx=18, pady=(0, 14))

        cols = ('№', 'Цена, тыс.', 'Продаж/мес')
        self.tree = ttk.Treeview(tf, columns=cols, show='headings', height=10)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview',
                        background=CARD, fieldbackground=CARD,
                        foreground=TEXT, rowheight=26,
                        font=('Segoe UI', 10), borderwidth=0)
        style.configure('Treeview.Heading',
                        background='#f1f5f9', foreground=TEXT,
                        font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.map('Treeview', background=[('selected', ACCENT)],
                  foreground=[('selected', 'white')])
        style.map('Treeview.Heading', background=[('active', '#e2e8f0')])

        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor=tk.CENTER, width=100)

        self.tree.pack(fill=tk.X)

    def mkbtn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg='white',
                         font=('Segoe UI', 10, 'bold'),
                         relief=tk.FLAT, bd=0,
                         activebackground=color,
                         activeforeground='white',
                         cursor='hand2',
                         padx=14, pady=8)

    def mkentry(self, parent, label, default=""):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill=tk.X, pady=4)
        tk.Label(row, text=label, bg=CARD, fg=MUTED,
                 width=14, anchor='w',
                 font=('Segoe UI', 10)).pack(side=tk.LEFT)
        e = tk.Entry(row, font=('Segoe UI', 11), bg='#f1f5f9',
                     fg=TEXT, relief=tk.FLAT, bd=0,
                     insertbackground=TEXT)
        if default:
            e.insert(0, default)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(8, 0))
        return e

    def build_test_form(self, parent):
        _, inner = self.card(
            parent,
            "Товар для распознавания",
            "Введите характеристики нового товара — алгоритм определит, "
            "к какому из найденных кластеров он относится."
        )

        self.te1 = self.mkentry(inner, "Цена (тыс.):", "30.0")
        self.te2 = self.mkentry(inner, "Продаж/мес:", "200.0")

    def build_actions(self, parent):
        _, inner = self.card(
            parent,
            "Кластеризация",
            "Нажмите кнопку — алгоритм разобьёт товары на кластеры "
            "и покажет, куда попал товар для распознавания."
        )

        self.mkbtn(inner, "▶  Метод K-средних", self.run_kmeans, ACCENT)\
            .pack(fill=tk.X, pady=(0, 6))
        self.mkbtn(inner, "▶  Метод максимина", self.run_maximin, RED)\
            .pack(fill=tk.X, pady=6)
        self.mkbtn(inner, "▶  Оба метода", self.run_both, PURPLE)\
            .pack(fill=tk.X, pady=(6, 0))

    def build_plots(self, parent):
        pf = tk.Frame(parent, bg=BG)
        pf.pack(fill=tk.BOTH, expand=True)

        self.fig1 = Figure(figsize=(5, 5), facecolor=BG, dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        c1 = FigureCanvasTkAgg(self.fig1, pf)
        c1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.canvas1 = c1

        self.fig2 = Figure(figsize=(5, 5), facecolor=BG, dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        c2 = FigureCanvasTkAgg(self.fig2, pf)
        c2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))
        self.canvas2 = c2

    def build_info(self, parent):
        wrapper = tk.Frame(parent, bg=CARD)
        wrapper.pack(fill=tk.X, pady=(12, 0))

        self.info = tk.Text(wrapper, height=6, font=('Consolas', 10),
                            bg=CARD, fg=TEXT, relief=tk.FLAT, bd=0,
                            padx=18, pady=14, wrap=tk.WORD)
        self.info.pack(fill=tk.X)
        self.info.insert(tk.END,
                         "Введите товар для распознавания и нажмите кнопку кластеризации.")
        self.info.config(state=tk.DISABLED)

    def set_info(self, text):
        self.info.config(state=tk.NORMAL)
        self.info.delete('1.0', tk.END)
        self.info.insert(tk.END, text)
        self.info.config(state=tk.DISABLED)

    def refresh_table(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for i, row in enumerate(self.dm.data):
            self.tree.insert('', tk.END,
                             values=(i + 1, f"{row[0]:.2f}", f"{row[1]:.0f}"))

    def read_test_point(self):
        try:
            f1 = float(self.te1.get())
            f2 = float(self.te2.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числа")
            return None
        self.dm.set_test_point(f1, f2)
        return self.dm.test_point

    def style_ax(self, ax, title):
        ax.clear()
        ax.set_title(title, fontsize=14, fontweight='bold', color=TEXT, pad=14)
        ax.set_xlabel('Цена (тыс. руб.)', fontsize=11, color=TEXT, labelpad=8)
        ax.set_ylabel('Продаж в месяц', fontsize=11, color=TEXT, labelpad=8)
        ax.grid(True, alpha=0.25, linestyle='--', color='#cbd5e1')
        ax.set_facecolor('#ffffff')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color('#e2e8f0')
            spine.set_linewidth(1.2)
        ax.tick_params(colors=MUTED, labelsize=10, length=4)

    def draw_initial(self):
        self.style_ax(self.ax1, 'Метод K-средних')
        self.style_ax(self.ax2, 'Метод максимина')
        self.fig1.tight_layout()
        self.fig2.tight_layout()
        self.canvas1.draw()
        self.canvas2.draw()

    def plot_clusters(self, ax, data, labels, centroids, title, test_label=None):
        self.style_ax(ax, title)

        for i in range(len(data)):
            c = CLUSTER_COLORS[int(labels[i]) % len(CLUSTER_COLORS)]
            ax.scatter(data[i, 0], data[i, 1], c=c, s=140, alpha=0.9,
                       edgecolors='white', linewidths=1.8, zorder=3)
            ax.annotate(str(i + 1), (data[i, 0], data[i, 1]),
                        fontsize=8, ha='center', va='center',
                        color='white', fontweight='bold', zorder=4)

        if centroids is not None and len(centroids) > 0:
            for i, c in enumerate(centroids):
                col = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                ax.scatter(c[0], c[1], c=col, s=420, marker='*',
                           edgecolors='white', linewidths=2, zorder=5)

        tp = self.dm.test_point
        if tp is not None:
            col = CLUSTER_COLORS[test_label % len(CLUSTER_COLORS)] \
                if test_label is not None else '#111827'
            ax.scatter(tp[0], tp[1], c=col, s=280, marker='X',
                       edgecolors='white', linewidths=2.2, zorder=6)

        handles = []
        labels_txt = []
        n_clusters = len(centroids) if centroids is not None else 0
        for i in range(n_clusters):
            col = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            name = get_cluster_name(centroids[i])
            handles.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=col, markeredgecolor='white',
                                  markersize=10, linestyle='None'))
            labels_txt.append(f'{name} (кластер {i+1})')

        handles.append(Line2D([0], [0], marker='*', color='w',
                              markerfacecolor='#64748b', markeredgecolor='white',
                              markersize=15, linestyle='None'))
        labels_txt.append('Центр кластера')

        if tp is not None:
            handles.append(Line2D([0], [0], marker='X', color='w',
                                  markerfacecolor='#111827', markeredgecolor='white',
                                  markersize=11, linestyle='None'))
            labels_txt.append('Товар для распознавания')

        ax.legend(handles, labels_txt,
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.18),
                  ncol=2,
                  frameon=False,
                  fontsize=9,
                  handletextpad=0.6,
                  columnspacing=1.2)

        ax.relim()
        ax.autoscale_view()
        ax.margins(x=0.08, y=0.08)

    def run_kmeans(self):
        tp = self.read_test_point()
        if tp is None:
            return

        raw_data = self.dm.data
        mean = raw_data.mean(axis=0)
        std = raw_data.std(axis=0)
        std[std == 0] = 1.0

        scaled_data = (raw_data - mean) / std
        scaled_tp = (tp - mean) / std

        labels = self.kmeans.fit(scaled_data)
        test_label = self.kmeans.predict(scaled_tp)

        unscaled_centroids = self.kmeans.centroids * std + mean

        labels, unscaled_centroids = reorder_clusters(labels, unscaled_centroids)

        test_label = int(np.argmin(np.linalg.norm(unscaled_centroids - tp, axis=1)))

        self.plot_clusters(self.ax1, raw_data, labels,
                           unscaled_centroids,
                           'Метод K-средних (со стандартизацией)', test_label)
        self.fig1.tight_layout()
        self.canvas1.draw()

        text = "МЕТОД K-СРЕДНИХ (Данные стандартизированы)\n"
        text += f"Найдено кластеров: {len(unscaled_centroids)}  |  итераций: {self.kmeans.n_iter_}\n\n"
        for i in range(len(unscaled_centroids)):
            members = [j + 1 for j, l in enumerate(labels) if l == i]
            name = get_cluster_name(unscaled_centroids[i])
            text += f"  • Кластер {i + 1} ({name}): товары {members}\n"
        text += (f"\nТовар для распознавания (цена {tp[0]:.0f} тыс., "
                 f"продаж {tp[1]:.0f}/мес) → кластер {test_label + 1}")
        self.set_info(text)

    def run_maximin(self):
        tp = self.read_test_point()
        if tp is None:
            return

        raw_data = self.dm.data
        mean = raw_data.mean(axis=0)
        std = raw_data.std(axis=0)
        std[std == 0] = 1.0

        scaled_data = (raw_data - mean) / std
        scaled_tp = (tp - mean) / std

        labels = self.maximin.fit(scaled_data)

        unscaled_centroids = self.maximin.centroids * std + mean

        labels, unscaled_centroids = reorder_clusters(labels, unscaled_centroids)

        test_label = int(np.argmin(np.linalg.norm(unscaled_centroids - tp, axis=1)))

        self.plot_clusters(self.ax2, raw_data, labels,
                           unscaled_centroids,
                           'Метод максимина (со стандартизацией)', test_label)
        self.fig2.tight_layout()
        self.canvas2.draw()

        text = "МЕТОД МАКСИМИНА (Данные стандартизированы)\n"
        text += f"Найдено кластеров: {len(unscaled_centroids)}\n\n"
        for i in range(len(unscaled_centroids)):
            members = [j + 1 for j, l in enumerate(labels) if l == i]
            name = get_cluster_name(unscaled_centroids[i])
            text += f"  • Кластер {i + 1} ({name}): товары {members}\n"
        text += (f"\nТовар для распознавания (цена {tp[0]:.0f} тыс., "
                 f"продаж {tp[1]:.0f}/мес) → кластер {test_label + 1}")
        self.set_info(text)

    def run_both(self):
        self.run_kmeans()
        self.run_maximin()