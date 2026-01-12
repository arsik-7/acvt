import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from enum import Enum
from itertools import product
import math
import json

PALETTE = {
    "gate": {
        "INPUT": "#FDE68A",
        "AND": "#BBF7D0",
        "NAND": "#86EFAC",
        "OR": "#BAE6FD",
        "NOR": "#C4B5FD",
        "NOT": "#F9A8D4",
        "XOR": "#FCD34D",
        "OUTPUT": "#FBCFE8",
    },
    "outline_on": "#0F766E",
    "outline_off": "#B91C1C",
    "shadow": "#C7D2FE",
    "wire_on": "#22C55E",
    "wire_off": "#94A3B8",
    "grid": "#E2E8F0",
    "canvas_bg": "#F8FAFC",
}

GATE_W, GATE_H = 95, 58
PIN_R = 5


class GateType(Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    AND = "AND"
    NAND = "NAND"
    OR = "OR"
    NOR = "NOR"
    NOT = "NOT"
    XOR = "XOR"


class Gate:
    _auto_id = 0

    def __init__(self, gate_type: GateType, x: int, y: int):
        self.type = gate_type
        self.x = x
        self.y = y
        self.inputs_count = 1 if gate_type in (GateType.INPUT, GateType.NOT, GateType.OUTPUT) else 2
        self.input_values = [0] * self.inputs_count
        self.output_value = 0
        Gate._auto_id += 1
        self.label = f"{gate_type.value}_{Gate._auto_id}"

    def body_bbox(self):
        return (self.x - GATE_W / 2, self.y - GATE_H / 2,
                self.x + GATE_W / 2, self.y + GATE_H / 2)

    def input_positions(self):
        step = GATE_H / (self.inputs_count + 1)
        return [
            (self.x - GATE_W / 2, self.y - GATE_H / 2 + step * (i + 1))
            for i in range(self.inputs_count)
        ]

    def output_position(self):
        return (self.x + GATE_W / 2, self.y)

    def toggle(self):
        if self.type == GateType.INPUT:
            self.output_value = 1 - self.output_value

    def evaluate(self):
        if self.type == GateType.INPUT:
            return
        if self.type == GateType.AND:
            self.output_value = 1 if all(self.input_values) else 0
        elif self.type == GateType.NAND:
            self.output_value = 0 if all(self.input_values) else 1
        elif self.type == GateType.OR:
            self.output_value = 1 if any(self.input_values) else 0
        elif self.type == GateType.NOR:
            self.output_value = 0 if any(self.input_values) else 1
        elif self.type == GateType.NOT:
            self.output_value = 1 - self.input_values[0]
        elif self.type == GateType.XOR:
            self.output_value = sum(self.input_values) % 2
        elif self.type == GateType.OUTPUT:
            self.output_value = self.input_values[0]


class Connection:
    def __init__(self, src: Gate, dst: Gate, dst_pin: int):
        self.src = src
        self.dst = dst
        self.dst_pin = dst_pin
        self.canvas_id = None


class TruthTableWindow(tk.Toplevel):
    def __init__(self, master, headers, rows):
        super().__init__(master)
        self.title("Таблица истинности")
        self.geometry("560x360")
        self.resizable(False, False)

        tree = ttk.Treeview(self, columns=headers, show="headings", selectmode="none")
        style = ttk.Style(self)
        style.configure("Treeview.Heading", font=("Inter", 11, "bold"))
        style.configure("Treeview", font=("Inter", 10))
        style.map("Treeview", background=[("selected", "#1E40AF")], foreground=[("selected", "white")])

        for head in headers:
            tree.heading(head, text=head)
            tree.column(head, width=70, anchor="center")

        vsb = ttk.Scrollbar(self, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        for row in rows:
            tree.insert("", "end", values=row)


class LogicSimulator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digital Logic Studio")
        self.root.minsize(1200, 720)

        self.gates: list[Gate] = []
        self.connections: list[Connection] = []
        self.selected_gate: Gate | None = None
        self.dragged_gate: Gate | None = None
        self.drag_offset = (0, 0)
        self.pending_connection: Gate | None = None
        self.temp_line_id = None

        self._build_ui()
        self._bind_canvas()
        self.draw_grid()
        self.refresh()
        self.root.mainloop()


    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(12, 8))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Digital Logic Studio", font=("Inter", 18, "bold")).pack(side=tk.LEFT)
        ttk.Button(top, text="Сохранить проект", command=self.save_project).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="Загрузить проект", command=self.load_project).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="Симулировать", command=self.simulate, style="Accent.TButton").pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="Таблица истинности", command=self.show_truth_table).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="Очистить поле", command=self.reset_workspace).pack(side=tk.RIGHT, padx=5)

        wrapper = tk.Frame(self.root)
        wrapper.pack(fill=tk.BOTH, expand=True)

        library = tk.Frame(wrapper, width=220, bg="#0F172A")
        library.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(library, text="Библиотека", fg="#E0E7FF", bg="#0F172A",
                 font=("Inter", 14, "bold")).pack(pady=(12, 6))
        for gate_type in (
            GateType.INPUT, GateType.AND, GateType.NAND, GateType.OR,
            GateType.NOR, GateType.NOT, GateType.XOR, GateType.OUTPUT
        ):
            tk.Button(
                library, text=gate_type.value, font=("Inter", 11),
                fg="#0F172A", bg="#E0E7FF", bd=0, relief=tk.FLAT,
                command=lambda gt=gate_type: self.create_gate(gt)
            ).pack(fill=tk.X, padx=18, pady=4)

        tk.Label(
            library,
            text="Как соединять:\n1) Клик по выводу\n2) Ведите линию к входу\n3) Отпустите или щёлкните по входу.\n\n"
                 "Как перемещать:\nпросто перетащите за корпус.\n\n"
                 "Двойной клик по INPUT переключает 0 ↔ 1.",
            fg="#F8FAFC", bg="#0F172A", justify="left", font=("Inter", 10)
        ).pack(fill=tk.X, padx=18, pady=18)

        canvas_holder = tk.Frame(wrapper, bg="#E5E7EB", padx=6, pady=6)
        canvas_holder.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_holder, bg=PALETTE["canvas_bg"],
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        info = tk.Frame(wrapper, width=260, bg="#F8FAFC", padx=12, pady=10)
        info.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(info, text="Инспектор элемента", font=("Inter", 13, "bold")).pack(anchor="w")
        self.info_var = tk.StringVar(value="Ничего не выбрано")
        ttk.Label(info, textvariable=self.info_var, justify="left",
                  font=("Inter", 10)).pack(fill=tk.X, pady=6)

        status = ttk.Frame(self.root, padding=(10, 4))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Готово. Соедините вывод с входом, перетаскивая мышью.")
        self.summary_var = tk.StringVar(value="IN: – | OUT: –")
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, anchor="w")
        ttk.Label(status, textvariable=self.summary_var).pack(side=tk.RIGHT, anchor="e")

    def _bind_canvas(self):
        self.canvas.bind("<Configure>", lambda _: self.draw_grid())
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<Double-Button-1>", self.canvas_double_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.canvas.bind("<Button-3>", self.canvas_right_click)


    def draw_grid(self):
        self.canvas.delete("grid")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        step = 28
        for x in range(0, w, step):
            self.canvas.create_line(x, 0, x, h, fill=PALETTE["grid"], tags="grid")
        for y in range(0, h, step):
            self.canvas.create_line(0, y, w, y, fill=PALETTE["grid"], tags="grid")
        self.canvas.tag_lower("grid")

    def create_gate(self, gate_type: GateType):
        gate = Gate(
            gate_type,
            x=320 + (len(self.gates) % 4) * 120,
            y=160 + (len(self.gates) // 4) * 110
        )
        if gate.type == GateType.INPUT:
            gate.output_value = 0
        self.gates.append(gate)
        self.refresh()

    def reset_workspace(self):
        if messagebox.askyesno("Очистить", "Удалить все элементы и соединения?"):
            self.gates.clear()
            self.connections.clear()
            self.selected_gate = None
            self.pending_connection = None
            self._remove_temp_line()
            self.refresh()
            self._update_summary()


    def canvas_click(self, event):
        gate, zone, pin = self.pick_gate_region(event.x, event.y)
        self.selected_gate = gate
        self._update_inspector()

        if gate and zone == "body":
            self.dragged_gate = gate
            self.drag_offset = (event.x - gate.x, event.y - gate.y)
            return

        if gate and zone == "output":
            self._start_connection(gate, event.x, event.y)
            return

        if gate and zone == "input" and self.pending_connection:
            self._finish_connection(gate, pin)
            return

        if not gate:
            self.pending_connection = None
            self._remove_temp_line()

    def canvas_double_click(self, event):
        gate, zone, _ = self.pick_gate_region(event.x, event.y)
        if gate and gate.type == GateType.INPUT and zone == "body":
            gate.toggle()
            self.simulate()

    def canvas_drag(self, event):
        if self.dragged_gate:
            dx, dy = self.drag_offset
            self.dragged_gate.x = event.x - dx
            self.dragged_gate.y = event.y - dy
            self.refresh()
        elif self.pending_connection and self.temp_line_id:
            sx, sy = self.pending_connection.output_position()
            self.canvas.coords(self.temp_line_id, sx, sy, event.x, event.y)

    def canvas_release(self, event):
        if self.pending_connection:
            gate, zone, pin = self.pick_gate_region(event.x, event.y)
            if gate and zone == "input":
                self._finish_connection(gate, pin)
            else:
                self.pending_connection = None
                self._remove_temp_line()
                self.status_var.set("Соединение отменено")
        self.dragged_gate = None

    def canvas_right_click(self, event):
        gate, _, _ = self.pick_gate_region(event.x, event.y)
        if gate:
            self.connections = [c for c in self.connections if c.src != gate and c.dst != gate]
            self.gates.remove(gate)
            if self.selected_gate == gate:
                self.selected_gate = None
            self.refresh()
            self._update_inspector()
            self._update_summary()
        else:
            conn = self.pick_connection(event.x, event.y)
            if conn:
                self.connections.remove(conn)
                self.refresh()
                self._update_summary()

    def _start_connection(self, gate, x, y):
        self.pending_connection = gate
        sx, sy = gate.output_position()
        self._remove_temp_line()
        self.temp_line_id = self.canvas.create_line(
            sx, sy, x, y, dash=(5, 3), width=2,
            fill="#2563EB", tags="content"
        )
        self.status_var.set("Ведите линию к входу и отпустите")

    def _finish_connection(self, gate, pin):
        if self.pending_connection and gate != self.pending_connection:
            if any(c.dst == gate and c.dst_pin == pin for c in self.connections):
                messagebox.showwarning("Соединение", "У этого входа уже есть провод.")
            else:
                self.connections.append(Connection(self.pending_connection, gate, pin))
                self.status_var.set("Соединение добавлено")
                self.simulate()
        self.pending_connection = None
        self._remove_temp_line()

    def _remove_temp_line(self):
        if self.temp_line_id:
            self.canvas.delete(self.temp_line_id)
        self.temp_line_id = None


    def pick_gate_region(self, x, y):
        for gate in reversed(self.gates):
            x0, y0, x1, y1 = gate.body_bbox()
            if x0 <= x <= x1 and y0 <= y <= y1:
                return gate, "body", None
            ox, oy = gate.output_position()
            if math.dist((x, y), (ox, oy)) <= PIN_R * 1.8:
                return gate, "output", None
            for idx, (ix, iy) in enumerate(gate.input_positions()):
                if math.dist((x, y), (ix, iy)) <= PIN_R * 1.8:
                    return gate, "input", idx
        return None, None, None

    def pick_connection(self, x, y, tolerance=5):
        for conn in self.connections:
            sx, sy = conn.src.output_position()
            dx, dy = conn.dst.input_positions()[conn.dst_pin]
            if self._distance_to_segment(x, y, sx, sy, dx, dy) <= tolerance:
                return conn
        return None

    @staticmethod
    def _distance_to_segment(px, py, x1, y1, x2, y2):
        line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if line_len_sq == 0:
            return math.dist((px, py), (x1, y1))
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        return math.dist((px, py), (proj_x, proj_y))


    def simulate(self, refresh_ui=True):
        for gate in self.gates:
            gate.input_values = [0] * gate.inputs_count
            if gate.type == GateType.INPUT:
                gate.evaluate()

        changed = True
        loops = 0
        while changed and loops < 60:
            loops += 1
            changed = False
            for conn in self.connections:
                conn.dst.input_values[conn.dst_pin] = conn.src.output_value
            for gate in self.gates:
                if gate.type != GateType.INPUT:
                    prev = gate.output_value
                    gate.evaluate()
                    if gate.output_value != prev:
                        changed = True

        if refresh_ui:
            self.refresh()
            self._update_summary()


    def show_truth_table(self):
        inputs = [g for g in self.gates if g.type == GateType.INPUT]
        outputs = [g for g in self.gates if g.type == GateType.OUTPUT]

        if not inputs or not outputs:
            messagebox.showinfo("Таблица", "Нужны хотя бы один вход и один выход.")
            return

        headers = [f"IN{i}" for i in range(len(inputs))] + [f"OUT{j}" for j in range(len(outputs))]
        rows = []

        saved_inputs = [g.output_value for g in inputs]

        for combo in product([0, 1], repeat=len(inputs)):
            for gate, bit in zip(inputs, combo):
                gate.output_value = bit
            self.simulate(refresh_ui=False)
            rows.append(combo + tuple(out.output_value for out in outputs))

        for gate, bit in zip(inputs, saved_inputs):
            gate.output_value = bit
        self.simulate(refresh_ui=True)

        TruthTableWindow(self.root, headers, rows)


    def save_project(self):
        if not self.gates:
            messagebox.showinfo("Сохранение", "Схема пуста — сохранять нечего.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON файлы", "*.json")],
            title="Сохранить проект"
        )
        if not filename:
            return

        gate_map = {id(gate): idx for idx, gate in enumerate(self.gates)}
        data = {
            "gates": [
                {
                    "type": gate.type.value,
                    "x": gate.x,
                    "y": gate.y,
                    "output": gate.output_value
                }
                for gate in self.gates
            ],
            "connections": [
                {
                    "source_index": gate_map[id(conn.src)],
                    "target_index": gate_map[id(conn.dst)],
                    "target_pin": conn.dst_pin
                }
                for conn in self.connections
            ]
        }

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Сохранение", "Проект успешно сохранён.")
        except OSError as exc:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{exc}")

    def load_project(self):
        filename = filedialog.askopenfilename(
            defaultextension=".json", filetypes=[("JSON файлы", "*.json")],
            title="Загрузить проект"
        )
        if not filename:
            return

        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Ошибка", f"Не удалось загрузить проект:\n{exc}")
            return

        if not isinstance(data, dict) or "gates" not in data or "connections" not in data:
            messagebox.showerror("Ошибка", "Неверный формат файла.")
            return

        if self.gates and not messagebox.askyesno("Загрузка", "Текущая схема будет удалена. Продолжить?"):
            return

        self.gates.clear()
        self.connections.clear()
        Gate._auto_id = 0

        for g_data in data["gates"]:
            try:
                gate_type = GateType(g_data["type"])
                gate = Gate(gate_type, g_data["x"], g_data["y"])
                gate.output_value = g_data.get("output", 0)
                self.gates.append(gate)
            except (KeyError, ValueError):
                messagebox.showerror("Ошибка", "Некорректные данные о гейте.")
                self.gates.clear()
                self.connections.clear()
                return

        for c_data in data["connections"]:
            try:
                src = self.gates[c_data["source_index"]]
                dst = self.gates[c_data["target_index"]]
                pin = c_data["target_pin"]
                if pin < dst.inputs_count:
                    self.connections.append(Connection(src, dst, pin))
            except (KeyError, IndexError):
                continue

        self.refresh()
        self.simulate()
        messagebox.showinfo("Загрузка", "Проект загружен.")


    def refresh(self):
        self.canvas.delete("content")
        for conn in self.connections:
            self._draw_connection(conn)
        for gate in self.gates:
            self._draw_gate(gate)
        self.canvas.tag_lower("grid")

    def _draw_gate(self, gate: Gate):
        x0, y0, x1, y1 = gate.body_bbox()
        self.canvas.create_rectangle(x0 + 5, y0 + 5, x1 + 5, y1 + 5,
                                     fill=PALETTE["shadow"], outline="", tags="content")
        outline = "#2563EB" if gate == self.selected_gate else (
            PALETTE["outline_on"] if gate.output_value else PALETTE["outline_off"]
        )
        self._rounded_rect(
            x0, y0, x1, y1, radius=16,
            fill=PALETTE["gate"].get(gate.type.value, "#E5E7EB"),
            outline=outline, width=2, tags="content"
        )

        label = gate.type.value if gate.type not in (GateType.INPUT, GateType.OUTPUT) \
            else f"{gate.type.value}\n{gate.output_value}"
        self.canvas.create_text(
            gate.x, gate.y, text=label, font=("Inter", 11, "bold"), tags="content"
        )

        for ix, iy in gate.input_positions():
            self.canvas.create_oval(ix - PIN_R, iy - PIN_R, ix + PIN_R, iy + PIN_R,
                                    fill="#111827", outline="", tags="content")
        ox, oy = gate.output_position()
        self.canvas.create_oval(ox - PIN_R, oy - PIN_R, ox + PIN_R, oy + PIN_R,
                                fill="#111827", outline="", tags="content")

        self.canvas.create_oval(
            ox - 8, y0 - 14, ox + 8, y0 + 2,
            fill=PALETTE["wire_on"] if gate.output_value else PALETTE["wire_off"],
            outline="", tags="content"
        )

    def _draw_connection(self, conn: Connection):
        sx, sy = conn.src.output_position()
        dx, dy = conn.dst.input_positions()[conn.dst_pin]
        mid_x = (sx + dx) / 2
        color = PALETTE["wire_on"] if conn.src.output_value else PALETTE["wire_off"]
        conn.canvas_id = self.canvas.create_line(
            sx, sy, mid_x, sy, mid_x, dy, dx, dy,
            smooth=True, splinesteps=24,
            width=3, fill=color, capstyle=tk.ROUND, tags="content"
        )

    def _rounded_rect(self, x0, y0, x1, y1, radius=12, **kwargs):
        points = [
            x0 + radius, y0,
            x1 - radius, y0,
            x1, y0,
            x1, y0 + radius,
            x1, y1 - radius,
            x1, y1,
            x1 - radius, y1,
            x0 + radius, y1,
            x0, y1,
            x0, y1 - radius,
            x0, y0 + radius,
            x0, y0
        ]
        return self.canvas.create_polygon(points, smooth=True, **kwargs)


    def _update_inspector(self):
        gate = self.selected_gate
        if not gate:
            self.info_var.set("Ничего не выбрано")
            return
        info = (
            f"Имя: {gate.label}\n"
            f"Тип: {gate.type.value}\n"
            f"Позиция: ({int(gate.x)}, {int(gate.y)})\n"
            f"Входов: {gate.inputs_count}\n"
            f"Выход: {gate.output_value}"
        )
        self.info_var.set(info)

    def _update_summary(self):
        inputs = [g for g in self.gates if g.type == GateType.INPUT]
        outputs = [g for g in self.gates if g.type == GateType.OUTPUT]
        in_text = ", ".join(f"{idx}={gate.output_value}" for idx, gate in enumerate(inputs)) or "–"
        out_text = ", ".join(f"{idx}={gate.output_value}" for idx, gate in enumerate(outputs)) or "–"
        self.summary_var.set(f"IN: {in_text} | OUT: {out_text}")


if __name__ == "__main__":
    LogicSimulator()