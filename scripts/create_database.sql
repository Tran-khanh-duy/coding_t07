/* ============================================================
   FACE ATTENDANCE DATABASE
============================================================ */

SET NOCOUNT ON
GO

-- ============================================================
-- 1. TAO DATABASE
-- ============================================================

IF DB_ID('FaceAttendanceDB') IS NULL
    CREATE DATABASE FaceAttendanceDB COLLATE Vietnamese_CI_AS
GO

USE FaceAttendanceDB
GO

-- ============================================================
-- 2. TAO BANG
-- ============================================================

-- Classes
CREATE TABLE Classes (
    class_id      INT           IDENTITY(1,1) PRIMARY KEY,
    class_code    NVARCHAR(20)  NOT NULL UNIQUE,
    class_name    NVARCHAR(100) NOT NULL,
    teacher_name  NVARCHAR(100) NULL,
    academic_year NVARCHAR(20)  NULL,
    is_active     BIT           NOT NULL DEFAULT 1,
    created_at    DATETIME2     NOT NULL DEFAULT GETDATE()
)
GO

-- Students
CREATE TABLE Students (
    student_id    INT           IDENTITY(1,1) PRIMARY KEY,
    student_code  NVARCHAR(20)  NOT NULL UNIQUE,
    full_name     NVARCHAR(100) NOT NULL,
    gender        NVARCHAR(10)  NULL,
    date_of_birth DATE          NULL,
    phone         NVARCHAR(20)  NULL,
    email         NVARCHAR(100) NULL,
    class_id      INT           NULL REFERENCES Classes(class_id),
    face_enrolled BIT           NOT NULL DEFAULT 0,
    created_at    DATETIME2     NOT NULL DEFAULT GETDATE()
)
GO

CREATE INDEX IX_Students_Class ON Students(class_id)
CREATE INDEX IX_Students_Code  ON Students(student_code)
GO

-- FaceEmbeddings
CREATE TABLE FaceEmbeddings (
    embedding_id     INT            IDENTITY(1,1) PRIMARY KEY,
    student_id       INT            NOT NULL REFERENCES Students(student_id) ON DELETE CASCADE,
    embedding_vector VARBINARY(MAX) NOT NULL,
    model_version    NVARCHAR(50)   NOT NULL DEFAULT 'buffalo_l',
    created_at       DATETIME2      NOT NULL DEFAULT GETDATE(),
    is_active        BIT            NOT NULL DEFAULT 1
)
GO

CREATE UNIQUE INDEX UX_Embedding_Active
    ON FaceEmbeddings(student_id)
    WHERE is_active = 1
GO

-- Cameras
CREATE TABLE Cameras (
    camera_id     INT           IDENTITY(1,1) PRIMARY KEY,
    camera_name   NVARCHAR(100) NOT NULL,
    location_desc NVARCHAR(200) NULL,
    rtsp_url      NVARCHAR(500) NULL,
    ip_address    NVARCHAR(50)  NULL,
    resolution    NVARCHAR(20)  NOT NULL DEFAULT '1280x720',
    is_active     BIT           NOT NULL DEFAULT 1
)
GO

-- AttendanceSessions
CREATE TABLE AttendanceSessions (
    session_id    INT           IDENTITY(1,1) PRIMARY KEY,
    session_code  NVARCHAR(50)  NOT NULL UNIQUE,
    class_id      INT           NOT NULL REFERENCES Classes(class_id),
    subject_name  NVARCHAR(100) NOT NULL,
    session_date  DATE          NOT NULL,
    start_time    DATETIME2     NULL,
    end_time      DATETIME2     NULL,
    status        NVARCHAR(20)  NOT NULL DEFAULT 'PENDING',
    present_count INT           NOT NULL DEFAULT 0,
    absent_count  INT           NOT NULL DEFAULT 0,
    created_at    DATETIME2     NOT NULL DEFAULT GETDATE()
)
GO

CREATE INDEX IX_Session_Class  ON AttendanceSessions(class_id)
CREATE INDEX IX_Session_Date   ON AttendanceSessions(session_date)
CREATE INDEX IX_Session_Status ON AttendanceSessions(status)
GO

-- AttendanceRecords
CREATE TABLE AttendanceRecords (
    record_id         INT           IDENTITY(1,1) PRIMARY KEY,
    session_id        INT           NOT NULL REFERENCES AttendanceSessions(session_id),
    student_id        INT           NOT NULL REFERENCES Students(student_id),
    check_in_time     DATETIME2     NULL,
    status            NVARCHAR(20)  NOT NULL DEFAULT 'ABSENT',
    recognition_score FLOAT         NULL,
    snapshot_path     NVARCHAR(500) NULL,
    camera_id         INT           NULL,
    created_at        DATETIME2     NOT NULL DEFAULT GETDATE()
)
GO

CREATE UNIQUE INDEX UX_Record_Session_Student ON AttendanceRecords(session_id, student_id)
CREATE INDEX IX_Record_Session ON AttendanceRecords(session_id)
CREATE INDEX IX_Record_Student ON AttendanceRecords(student_id)
CREATE INDEX IX_Record_Status  ON AttendanceRecords(status)
GO

PRINT '=> Tao bang hoan tat.'
GO

-- ============================================================
-- 3. STORED PROCEDURES
-- ============================================================

-- SP1: Load embeddings vao RAM
-- Python unpack: (student_id, student_code, full_name, embedding_vector)
CREATE OR ALTER PROCEDURE sp_GetAllEmbeddings
AS
BEGIN
    SET NOCOUNT ON
    SELECT
        s.student_id,
        s.student_code,
        s.full_name,
        fe.embedding_vector,
        s.class_id
    FROM FaceEmbeddings fe
    JOIN Students s ON s.student_id = fe.student_id
    WHERE fe.is_active = 1
    ORDER BY fe.student_id
END
GO

-- SP2: Ghi nhan diem danh
CREATE OR ALTER PROCEDURE sp_RecordAttendance
    @session_id INT,
    @student_id INT,
    @score      FLOAT
AS
BEGIN
    SET NOCOUNT ON
    MERGE AttendanceRecords AS T
    USING (SELECT @session_id AS sid, @student_id AS uid) AS S
        ON T.session_id = S.sid AND T.student_id = S.uid
    WHEN MATCHED THEN
        UPDATE SET
            status            = 'PRESENT',
            check_in_time     = GETDATE(),
            recognition_score = @score
    WHEN NOT MATCHED THEN
        INSERT (session_id, student_id, status, check_in_time, recognition_score)
        VALUES (@session_id, @student_id, 'PRESENT', GETDATE(), @score)
    ;

    UPDATE AttendanceSessions
    SET present_count = (
        SELECT COUNT(*) FROM AttendanceRecords
        WHERE session_id = @session_id AND status = 'PRESENT'
    )
    WHERE session_id = @session_id
END
GO

-- SP3: Bao cao buoi hoc
CREATE OR ALTER PROCEDURE sp_GetSessionReport
    @session_id INT
AS
BEGIN
    SET NOCOUNT ON
    SELECT
        s.student_code,
        s.full_name,
        c.class_name,
        ISNULL(ar.status, 'ABSENT') AS attendance_status,
        ar.check_in_time,
        ar.recognition_score,
        sess.session_date,
        sess.subject_name,
        sess.start_time
    FROM AttendanceSessions sess
    JOIN Students s ON s.class_id = sess.class_id
    JOIN Classes  c ON c.class_id = s.class_id
    LEFT JOIN AttendanceRecords ar
        ON ar.student_id = s.student_id
        AND ar.session_id = @session_id
    WHERE sess.session_id = @session_id
    ORDER BY
        CASE ISNULL(ar.status, 'ABSENT') WHEN 'PRESENT' THEN 0 ELSE 1 END,
        s.full_name
END
GO

PRINT '=> Stored procedures hoan tat.'
GO

-- ============================================================
-- 4. DU LIEU MAU
-- ============================================================

-- Classes
INSERT INTO Classes (class_code, class_name, teacher_name, academic_year)
VALUES
    (N'CNTT01', N'Lập trình Python Cơ bản',     N'Nguyễn Văn An',  N'2024-2025'),
    (N'CNTT02', N'Lập trình Web Frontend',       N'Trần Thị Bình',  N'2024-2025'),
    (N'CNTT03', N'Cơ sở Dữ liệu SQL Server',    N'Lê Minh Châu',   N'2024-2025'),
    (N'CNTT04', N'Trí tuệ Nhân tạo & ML',       N'Phạm Quốc Dũng', N'2024-2025'),
    (N'CNTT05', N'An toàn Thông tin & Bảo mật', N'Hoàng Thị Mai',  N'2024-2025')
GO

-- Students (10 hoc vien - 2 lop dau)
INSERT INTO Students (student_code, full_name, gender, date_of_birth, phone, email, class_id)
SELECT col, ten, gt, ngay, sdt, mail, (SELECT class_id FROM Classes WHERE class_code = lop)
FROM (VALUES
    (N'HV001', N'Trần Khánh Duy',   N'Nam', CAST('2003-05-15' AS DATE), N'0901234567', N'duy@gmail.com',   N'CNTT01'),
    (N'HV002', N'Nguyễn Thị Hương', N'Nữ',  CAST('2003-08-22' AS DATE), N'0912345678', N'huong@gmail.com', N'CNTT01'),
    (N'HV003', N'Lê Văn Minh',      N'Nam', CAST('2002-11-30' AS DATE), N'0923456789', N'minh@gmail.com',  N'CNTT01'),
    (N'HV004', N'Phạm Thị Lan',     N'Nữ',  CAST('2003-03-14' AS DATE), N'0934567890', N'lan@gmail.com',   N'CNTT01'),
    (N'HV005', N'Đỗ Quang Huy',     N'Nam', CAST('2002-07-08' AS DATE), N'0945678901', N'huy@gmail.com',   N'CNTT01'),
    (N'HV006', N'Vũ Thị Ngọc',      N'Nữ',  CAST('2003-01-20' AS DATE), N'0956789012', N'ngoc@gmail.com',  N'CNTT02'),
    (N'HV007', N'Hoàng Văn Phúc',   N'Nam', CAST('2002-09-05' AS DATE), N'0967890123', N'phuc@gmail.com',  N'CNTT02'),
    (N'HV008', N'Bùi Thị Quyên',    N'Nữ',  CAST('2003-12-18' AS DATE), N'0978901234', N'quyen@gmail.com', N'CNTT02'),
    (N'HV009', N'Ngô Thanh Sơn',    N'Nam', CAST('2002-04-27' AS DATE), N'0989012345', N'son@gmail.com',   N'CNTT02'),
    (N'HV010', N'Đinh Thị Thảo',    N'Nữ',  CAST('2003-06-11' AS DATE), N'0990123456', N'thao@gmail.com',  N'CNTT02')
) AS v(col, ten, gt, ngay, sdt, mail, lop)
GO

-- Cameras
INSERT INTO Cameras (camera_name, location_desc, rtsp_url, ip_address, resolution)
VALUES
    (N'Camera Sảnh Tầng 1',    N'Cửa vào chính',           N'rtsp://admin:admin@192.168.1.101:554/stream', N'192.168.1.101', N'1280x720'),
    (N'Camera Phòng Học A101', N'Phòng học tầng 2',         N'rtsp://admin:admin@192.168.1.102:554/stream', N'192.168.1.102', N'1280x720'),
    (N'Camera Phòng Học A102', N'Phòng học tầng 2',         N'rtsp://admin:admin@192.168.1.103:554/stream', N'192.168.1.103', N'1280x720'),
    (N'Camera Hành Lang T3',   N'Hành lang tầng 3',         N'rtsp://admin:admin@192.168.1.104:554/stream', N'192.168.1.104', N'1920x1080'),
    (N'Webcam USB',            N'Webcam kết nối trực tiếp',  NULL,                                          NULL,             N'1280x720')
GO

-- AttendanceSessions
INSERT INTO AttendanceSessions
    (session_code, class_id, subject_name, session_date, start_time, end_time, status, present_count, absent_count)
SELECT s.code, c.class_id, s.ten, s.ngay, s.bt, s.kt, s.tt, s.cm, s.vm
FROM Classes c
JOIN (VALUES
    (N'CNTT01-20260308-S1', N'CNTT01', N'Python - Buổi 1: Biến & Kiểu dữ liệu',
     CAST('2026-03-08' AS DATE), CAST('2026-03-08 08:00' AS DATETIME2), CAST('2026-03-08 10:00' AS DATETIME2), N'COMPLETED', 4, 1),

    (N'CNTT01-20260309-S2', N'CNTT01', N'Python - Buổi 2: Vòng lặp & Hàm',
     CAST('2026-03-09' AS DATE), CAST('2026-03-09 08:00' AS DATETIME2), CAST('2026-03-09 10:00' AS DATETIME2), N'COMPLETED', 5, 0),

    (N'CNTT01-20260310-S3', N'CNTT01', N'Python - Buổi 3: List & Dictionary',
     CAST('2026-03-10' AS DATE), CAST('2026-03-10 08:00' AS DATETIME2), CAST('2026-03-10 10:00' AS DATETIME2), N'COMPLETED', 3, 2),

    (N'CNTT01-20260311-S4', N'CNTT01', N'Python - Buổi 4: OOP',
     CAST('2026-03-11' AS DATE), NULL, NULL, N'PENDING', 0, 0),

    (N'CNTT02-20260309-S1', N'CNTT02', N'Web Frontend - Buổi 1: HTML & CSS',
     CAST('2026-03-09' AS DATE), CAST('2026-03-09 13:00' AS DATETIME2), CAST('2026-03-09 15:00' AS DATETIME2), N'COMPLETED', 5, 0)
) AS s(code, lop, ten, ngay, bt, kt, tt, cm, vm)
    ON c.class_code = s.lop
GO

-- AttendanceRecords
INSERT INTO AttendanceRecords (session_id, student_id, status, check_in_time, recognition_score)
SELECT se.session_id, st.student_id, r.tt, r.gio, r.diem
FROM (VALUES
    -- Buoi 1: 
    (N'CNTT01-20260308-S1', N'HV001', N'PRESENT', CAST('2026-03-08 08:02:11' AS DATETIME2), 0.943),
    (N'CNTT01-20260308-S1', N'HV002', N'PRESENT', CAST('2026-03-08 08:03:45' AS DATETIME2), 0.921),
    (N'CNTT01-20260308-S1', N'HV003', N'PRESENT', CAST('2026-03-08 08:01:30' AS DATETIME2), 0.967),
    (N'CNTT01-20260308-S1', N'HV004', N'PRESENT', CAST('2026-03-08 08:05:20' AS DATETIME2), 0.889),
    (N'CNTT01-20260308-S1', N'HV005', N'ABSENT',  NULL,                                      NULL),
    -- Buoi 2: 
    (N'CNTT01-20260309-S2', N'HV001', N'PRESENT', CAST('2026-03-09 08:01:05' AS DATETIME2), 0.956),
    (N'CNTT01-20260309-S2', N'HV002', N'PRESENT', CAST('2026-03-09 08:02:33' AS DATETIME2), 0.934),
    (N'CNTT01-20260309-S2', N'HV003', N'PRESENT', CAST('2026-03-09 08:00:58' AS DATETIME2), 0.978),
    (N'CNTT01-20260309-S2', N'HV004', N'PRESENT', CAST('2026-03-09 08:04:12' AS DATETIME2), 0.901),
    (N'CNTT01-20260309-S2', N'HV005', N'PRESENT', CAST('2026-03-09 08:03:47' AS DATETIME2), 0.918),
    -- Buoi 3: 
    (N'CNTT01-20260310-S3', N'HV001', N'PRESENT', CAST('2026-03-10 08:00:45' AS DATETIME2), 0.961),
    (N'CNTT01-20260310-S3', N'HV002', N'ABSENT',  NULL,                                      NULL),
    (N'CNTT01-20260310-S3', N'HV003', N'PRESENT', CAST('2026-03-10 08:02:18' AS DATETIME2), 0.947),
    (N'CNTT01-20260310-S3', N'HV004', N'ABSENT',  NULL,                                      NULL),
    (N'CNTT01-20260310-S3', N'HV005', N'PRESENT', CAST('2026-03-10 08:01:33' AS DATETIME2), 0.925),
    -- Buoi 4 
    (N'CNTT02-20260309-S1', N'HV006', N'PRESENT', CAST('2026-03-09 13:01:22' AS DATETIME2), 0.932),
    (N'CNTT02-20260309-S1', N'HV007', N'PRESENT', CAST('2026-03-09 13:00:38' AS DATETIME2), 0.955),
    (N'CNTT02-20260309-S1', N'HV008', N'PRESENT', CAST('2026-03-09 13:02:45' AS DATETIME2), 0.912),
    (N'CNTT02-20260309-S1', N'HV009', N'PRESENT', CAST('2026-03-09 13:03:11' AS DATETIME2), 0.948),
    (N'CNTT02-20260309-S1', N'HV010', N'PRESENT', CAST('2026-03-09 13:01:59' AS DATETIME2), 0.939)
) AS r(scode, stcode, tt, gio, diem)
JOIN AttendanceSessions se ON se.session_code = r.scode
JOIN Students           st ON st.student_code = r.stcode
GO

PRINT '=> Du lieu mau hoan tat.'
GO

-- ============================================================
-- 5. KIEM TRA
-- ============================================================

SELECT [Bang]          = 'Classes',            [So rows] = COUNT(*) FROM Classes            UNION ALL
SELECT                   'Students',                       COUNT(*) FROM Students            UNION ALL
SELECT                   'FaceEmbeddings',                 COUNT(*) FROM FaceEmbeddings      UNION ALL
SELECT                   'Cameras',                        COUNT(*) FROM Cameras             UNION ALL
SELECT                   'AttendanceSessions',              COUNT(*) FROM AttendanceSessions  UNION ALL
SELECT                   'AttendanceRecords',               COUNT(*) FROM AttendanceRecords
GO

PRINT '===== DATABASE READY ====='
GO