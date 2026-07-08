from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from src.services.coordinate_service import CoordinateService


class ManualInputDialog(QDialog):
    """OCR 실패 시 수동으로 지역/좌표 입력"""

    def __init__(
        self,
        coordinate_service: CoordinateService,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.coordinate_service = coordinate_service
        self.setWindowTitle("수동 입력")
        self.setMinimumWidth(320)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(
            "자동 인식에 실패했습니다.\n지역과 좌표를 직접 입력하세요."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self.zone_combo = QComboBox()
        for zone in self.coordinate_service.zones:
            self.zone_combo.addItem(zone.get("name_ko", zone["id"]), zone["id"])
        form.addRow("지역:", self.zone_combo)

        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(0, 42)
        self.x_spin.setDecimals(1)
        self.x_spin.setSingleStep(0.1)
        form.addRow("X:", self.x_spin)

        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(0, 42)
        self.y_spin.setDecimals(1)
        self.y_spin.setSingleStep(0.1)
        form.addRow("Y:", self.y_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> tuple[dict, float, float] | None:
        if self.exec() != QDialog.DialogCode.Accepted:
            return None
        zone_id = self.zone_combo.currentData()
        zone = self.coordinate_service.get_zone(zone_id)
        if not zone:
            return None
        return zone, self.x_spin.value(), self.y_spin.value()
