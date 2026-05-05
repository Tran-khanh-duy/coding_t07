CREATE DATABASE IF NOT EXISTS faceattendancedb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE faceattendancedb;

CREATE TABLE Classes (
    class_id      INT AUTO_INCREMENT PRIMARY KEY,
    class_code    VARCHAR(20)  NOT NULL UNIQUE,
    class_name    VARCHAR(100) NOT NULL,
    teacher_name  VARCHAR(100) NULL,
    academic_year VARCHAR(20)  NULL,
    is_active     BOOLEAN      NOT NULL DEFAULT 1,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Students (
    student_id    INT AUTO_INCREMENT PRIMARY KEY,
    student_code  VARCHAR(20)  NOT NULL UNIQUE,
    full_name     VARCHAR(100) NOT NULL,
    gender        VARCHAR(10)  NULL,
    date_of_birth DATE          NULL,
    phone         VARCHAR(20)  NULL,
    email         VARCHAR(100) NULL,
    class_id      INT           NULL,
    class_name    VARCHAR(100) NULL,
    building      VARCHAR(100) NULL,
    floor         VARCHAR(50)  NULL,
    room          VARCHAR(50)  NULL,
    face_enrolled BOOLEAN      NOT NULL DEFAULT 0,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (class_id) REFERENCES Classes(class_id)
);

CREATE INDEX IX_Students_Class ON Students(class_id);
CREATE INDEX IX_Students_Code  ON Students(student_code);

CREATE TABLE FaceEmbeddings (
    embedding_id     INT AUTO_INCREMENT PRIMARY KEY,
    student_id       INT            NOT NULL,
    embedding_vector LONGBLOB       NOT NULL,
    model_version    VARCHAR(50)    NOT NULL DEFAULT 'buffalo_l',
    created_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active        BOOLEAN        NOT NULL DEFAULT 1,
    FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE
);

-- Note: MySQL doesn't support partial indexes like WHERE is_active=1 directly in UNIQUE INDEX. 
-- For simplicity, we just index both columns.
CREATE INDEX UX_Embedding_Active ON FaceEmbeddings(student_id, is_active);

CREATE TABLE Cameras (
    camera_id     INT AUTO_INCREMENT PRIMARY KEY,
    camera_name   VARCHAR(100) NOT NULL,
    location_desc VARCHAR(200) NULL,
    rtsp_url      VARCHAR(500) NULL,
    ip_address    VARCHAR(50)  NULL,
    resolution    VARCHAR(20)  NOT NULL DEFAULT '1280x720',
    area_id       VARCHAR(100) NULL,
    is_active     BOOLEAN      NOT NULL DEFAULT 1
);

CREATE TABLE AttendanceSessions (
    session_id    INT AUTO_INCREMENT PRIMARY KEY,
    session_code  VARCHAR(50)  NOT NULL UNIQUE,
    class_id      INT           NOT NULL,
    subject_name  VARCHAR(100) NOT NULL,
    session_date  DATE          NOT NULL,
    start_time    DATETIME      NULL,
    end_time      DATETIME      NULL,
    status        VARCHAR(20)  NOT NULL DEFAULT 'PENDING',
    present_count INT           NOT NULL DEFAULT 0,
    absent_count  INT           NOT NULL DEFAULT 0,
    created_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (class_id) REFERENCES Classes(class_id)
);

CREATE INDEX IX_Session_Class  ON AttendanceSessions(class_id);
CREATE INDEX IX_Session_Date   ON AttendanceSessions(session_date);
CREATE INDEX IX_Session_Status ON AttendanceSessions(status);

CREATE TABLE AttendanceRecords (
    record_id         INT AUTO_INCREMENT PRIMARY KEY,
    session_id        INT           NOT NULL,
    student_id        INT           NOT NULL,
    check_in_time     DATETIME      NULL,
    status            VARCHAR(20)  NOT NULL DEFAULT 'ABSENT',
    recognition_score FLOAT         NULL,
    snapshot_path     VARCHAR(500) NULL,
    camera_id         INT           NULL,
    created_at        DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES AttendanceSessions(session_id),
    FOREIGN KEY (student_id) REFERENCES Students(student_id)
);

CREATE UNIQUE INDEX UX_Record_Session_Student ON AttendanceRecords(session_id, student_id);
CREATE INDEX IX_Record_Session ON AttendanceRecords(session_id);
CREATE INDEX IX_Record_Student ON AttendanceRecords(student_id);
CREATE INDEX IX_Record_Status  ON AttendanceRecords(status);

DELIMITER //

CREATE PROCEDURE sp_GetAllEmbeddings()
BEGIN
    SELECT
        s.student_id,
        s.student_code,
        s.full_name,
        fe.embedding_vector,
        s.class_id
    FROM FaceEmbeddings fe
    JOIN Students s ON s.student_id = fe.student_id
    WHERE fe.is_active = 1
    ORDER BY fe.student_id;
END //

CREATE PROCEDURE sp_RecordAttendance(
    IN p_session_id INT,
    IN p_student_id INT,
    IN p_score FLOAT
)
BEGIN
    INSERT INTO AttendanceRecords (session_id, student_id, status, check_in_time, recognition_score)
    VALUES (p_session_id, p_student_id, 'PRESENT', NOW(), p_score)
    ON DUPLICATE KEY UPDATE 
        status = 'PRESENT',
        check_in_time = NOW(),
        recognition_score = p_score;

    UPDATE AttendanceSessions
    SET present_count = (
        SELECT COUNT(*) FROM AttendanceRecords
        WHERE session_id = p_session_id AND status = 'PRESENT'
    )
    WHERE session_id = p_session_id;
END //

CREATE PROCEDURE sp_GetSessionReport(
    IN p_session_id INT
)
BEGIN
    SELECT
        s.student_code,
        s.full_name,
        c.class_name,
        IFNULL(ar.status, 'ABSENT') AS attendance_status,
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
        AND ar.session_id = p_session_id
    WHERE sess.session_id = p_session_id
    ORDER BY
        CASE WHEN IFNULL(ar.status, 'ABSENT') = 'PRESENT' THEN 0 ELSE 1 END,
        s.full_name;
END //

DELIMITER ;
