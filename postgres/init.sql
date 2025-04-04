-- Mobile Device Information
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
('iPhone SE', 'iOS 14', 2020),
('Galaxy S22', 'Android 13', 2022),
('iPhone 14 Pro', 'iOS 16', 2022),
('Pixel 7', 'Android 13', 2022),
('Galaxy A52', 'Android 11', 2021),
('iPhone 12', 'iOS 14', 2020),
('Pixel 5', 'Android 11', 2020),
('Galaxy Fold 3', 'Android 12', 2021),
('iPhone XR', 'iOS 13', 2019),
('Pixel 4a', 'Android 11', 2020),
('Galaxy S23', 'Android 14', 2023),
('iPhone 15', 'iOS 17', 2023),
('Pixel 8', 'Android 14', 2023),
('Galaxy Z Flip 4', 'Android 13', 2022),
('iPhone 13 Mini', 'iOS 15', 2021),
('Pixel 6a', 'Android 13', 2022),
('Galaxy M31', 'Android 10', 2020),
('iPhone 11', 'iOS 13', 2019),
('Pixel 3 XL', 'Android 10', 2019),
('Galaxy Note 10', 'Android 10', 2019),
('iPhone XS Max', 'iOS 12', 2018);

-- Manufacturers Table
CREATE TABLE manufacturers (
    manufacturer_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT
);

INSERT INTO manufacturers (name, country) VALUES
('Samsung', 'South Korea'),
('Apple', 'United States'),
('Google', 'United States');

-- Add manufacturer_id to devices
ALTER TABLE devices
ADD COLUMN manufacturer_id INTEGER REFERENCES manufacturers(manufacturer_id);

-- Update devices with manufacturer_id
UPDATE devices SET manufacturer_id = 1 WHERE model_name LIKE 'Galaxy%';
UPDATE devices SET manufacturer_id = 2 WHERE model_name LIKE 'iPhone%';
UPDATE devices SET manufacturer_id = 3 WHERE model_name LIKE 'Pixel%';
