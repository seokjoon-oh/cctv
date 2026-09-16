-- Reconstructed example matching save_to_db() INSERT columns.
-- This is not a dump of the original database. Use a new development database.
CREATE DATABASE IF NOT EXISTS capstone_db
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE capstone_db;

CREATE TABLE IF NOT EXISTS detection_log (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    test_group VARCHAR(64) NOT NULL,
    model_name VARCHAR(64) NOT NULL,
    image_name VARCHAR(512) NOT NULL,
    person_idx INT NOT NULL,
    confidence DOUBLE NOT NULL,
    final_result VARCHAR(32) NOT NULL,
    best_distance DOUBLE NULL,
    best_detector VARCHAR(64) NULL,
    best_crop_type VARCHAR(32) NULL,
    event_type VARCHAR(64) NOT NULL,
    alert_required BOOLEAN NOT NULL,
    alert_message TEXT NOT NULL,
    annotated_path TEXT NOT NULL,
    detected_at DATETIME NOT NULL,
    INDEX idx_detection_time (detected_at),
    INDEX idx_event_time (event_type, detected_at)
);
