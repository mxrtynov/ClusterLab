import tkinter as tk
from ClusterApp import ClusterApp


def main():
    root = tk.Tk()
    app = ClusterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()