-- Sample Data
CREATE TABLE devices (
    device_id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    os_version TEXT,
    release_year INTEGER
);

INSERT INTO devices (model_name, os_version, release_year) VALUES
('Galaxy S21', 'Android 12', 2021),
('iPhone 13', 'iOS 15', 2021),
('Pixel 6', 'Android 12', 2021),
('Galaxy Note 20', 'Android 11', 2020),
('iPhone SE', 'iOS 14', 2020);
