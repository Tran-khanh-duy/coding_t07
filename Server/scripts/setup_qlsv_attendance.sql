-- ================================================================
-- setup_qlsv_attendance.sql
-- Tích hợp Face Attendance vào database qlsv (chạy 1 lần)
-- Yêu cầu sẵn có: lop(IDLop, TenLop), hocvien(MaHV, HoTen, GioiTinh, SoDienThoai, IDLop)
-- ================================================================
USE qlsv;

-- ── 1. Mở rộng bảng hocvien ──────────────────────────────────────
-- Thêm cột id INT AUTO_INCREMENT để dùng làm khoá số nguyên
ALTER TABLE hocvien
    ADD COLUMN IF NOT EXISTS id            INT AUTO_INCREMENT UNIQUE,
    ADD COLUMN IF NOT EXISTS face_enrolled BOOLEAN      NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS email         VARCHAR(100) NULL,
    ADD COLUMN IF NOT EXISTS date_of_birth DATE         NULL,
    ADD COLUMN IF NOT EXISTS building      VARCHAR(100) NULL,
    ADD COLUMN IF NOT EXISTS floor         VARCHAR(50)  NULL,
    ADD COLUMN IF NOT EXISTS room          VARCHAR(50)  NULL,
    ADD COLUMN IF NOT EXISTS created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP;

CREATE INDEX IF NOT EXISTS IX_hocvien_IDLop ON hocvien(IDLop);

-- ── 2. FaceEmbeddings ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS FaceEmbeddings (
    embedding_id     INT AUTO_INCREMENT PRIMARY KEY,
    student_id       INT         NOT NULL,   -- tham chiếu hocvien.id
    embedding_vector LONGBLOB    NOT NULL,
    model_version    VARCHAR(50) NOT NULL DEFAULT 'buffalo_l',
    created_at       DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active        BOOLEAN     NOT NULL DEFAULT 1,
    FOREIGN KEY (student_id) REFERENCES hocvien(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS UX_Embedding_Active ON FaceEmbeddings(student_id, is_active);

-- ── 3. Cameras ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Cameras (
    camera_id     INT AUTO_INCREMENT PRIMARY KEY,
    camera_name   VARCHAR(100) NOT NULL,
    location_desc VARCHAR(200) NULL,
    rtsp_url      VARCHAR(500) NULL,
    ip_address    VARCHAR(50)  NULL,
    resolution    VARCHAR(20)  NOT NULL DEFAULT '1280x720',
    area_id       VARCHAR(100) NULL,
    is_active     BOOLEAN      NOT NULL DEFAULT 1
);

-- ── 4. AttendanceSessions ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS AttendanceSessions (
    session_id    INT AUTO_INCREMENT PRIMARY KEY,
    session_code  VARCHAR(50)  NOT NULL UNIQUE,
    class_id      VARCHAR(20)  NOT NULL,     -- tham chiếu lop.IDLop
    subject_name  VARCHAR(100) NOT NULL,
    session_date  DATE         NOT NULL,
    start_time    DATETIME     NULL,
    end_time      DATETIME     NULL,
    status        VARCHAR(20)  NOT NULL DEFAULT 'PENDING',
    present_count INT          NOT NULL DEFAULT 0,
    absent_count  INT          NOT NULL DEFAULT 0,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (class_id) REFERENCES lop(IDLop)
);
CREATE INDEX IF NOT EXISTS IX_Session_Lop    ON AttendanceSessions(class_id);
CREATE INDEX IF NOT EXISTS IX_Session_Date   ON AttendanceSessions(session_date);
CREATE INDEX IF NOT EXISTS IX_Session_Status ON AttendanceSessions(status);

-- ── 5. AttendanceRecords ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS AttendanceRecords (
    record_id         INT AUTO_INCREMENT PRIMARY KEY,
    session_id        INT         NOT NULL,
    student_id        INT         NOT NULL,   -- tham chiếu hocvien.id
    check_in_time     DATETIME    NULL,
    status            VARCHAR(20) NOT NULL DEFAULT 'ABSENT',
    recognition_score FLOAT       NULL,
    snapshot_path     VARCHAR(500) NULL,
    camera_id         INT         NULL,
    created_at        DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES AttendanceSessions(session_id),
    FOREIGN KEY (student_id) REFERENCES hocvien(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS UX_Record_Session_Student ON AttendanceRecords(session_id, student_id);
CREATE INDEX IF NOT EXISTS IX_Record_Session ON AttendanceRecords(session_id);
CREATE INDEX IF NOT EXISTS IX_Record_Student ON AttendanceRecords(student_id);
CREATE INDEX IF NOT EXISTS IX_Record_Status  ON AttendanceRecords(status);

-- ── 6. Stored Procedures ─────────────────────────────────────────
DROP PROCEDURE IF EXISTS sp_GetAllEmbeddings;
DROP PROCEDURE IF EXISTS sp_GetSessionReport;

DELIMITER //

CREATE PROCEDURE sp_GetAllEmbeddings()
BEGIN
    SELECT hv.id, hv.MaHV, hv.HoTen, fe.embedding_vector, hv.IDLop
    FROM FaceEmbeddings fe
    JOIN hocvien hv ON hv.id = fe.student_id
    WHERE fe.is_active = 1
    ORDER BY fe.student_id;
END //

CREATE PROCEDURE sp_GetSessionReport(IN p_session_id INT)
BEGIN
    SELECT
        hv.MaHV, hv.HoTen, l.TenLop,
        IFNULL(ar.status,'ABSENT'), ar.check_in_time,
        ar.recognition_score, sess.session_date,
        sess.subject_name, sess.start_time
    FROM AttendanceSessions sess
    JOIN hocvien hv ON hv.IDLop = sess.class_id
    JOIN lop     l  ON l.IDLop  = hv.IDLop
    LEFT JOIN AttendanceRecords ar
           ON ar.student_id = hv.id AND ar.session_id = p_session_id
    WHERE sess.session_id = p_session_id
    ORDER BY CASE WHEN IFNULL(ar.status,'ABSENT')='PRESENT' THEN 0 ELSE 1 END, hv.HoTen;
END //

DELIMITER ;
