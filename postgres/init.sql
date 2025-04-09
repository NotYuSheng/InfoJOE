-- Manufacturers Table
CREATE TABLE manufacturers (
    manufacturer_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT
);

-- Mobile Device Information
CREATE TABLE devices (
    device_id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    os_version TEXT,
    release_year INTEGER,
    price_usd INTEGER,
    battery_mah INTEGER,
    screen_size FLOAT,
    weight_g INTEGER,
    manufacturer_id INTEGER REFERENCES manufacturers(manufacturer_id)
);

INSERT INTO manufacturers (name, country) VALUES
('Samsung', 'South Korea'),
('Apple', 'United States'),
('Google', 'United States');

INSERT INTO devices (model_name, os_version, release_year, price_usd, battery_mah, screen_size, weight_g, manufacturer_id) VALUES
('Galaxy S21', 'Android 12', 2021, 799, 4000, 6.2, 169, 1),
('iPhone 13', 'iOS 15', 2021, 799, 3095, 6.1, 174, 2),
('Pixel 6', 'Android 12', 2021, 599, 4614, 6.4, 207, 3),
('Galaxy Note 20', 'Android 11', 2020, 999, 4300, 6.7, 192, 1),
('iPhone SE', 'iOS 14', 2020, 399, 1821, 4.7, 148, 2),
('Galaxy S22', 'Android 13', 2022, 799, 4500, 6.6, 168, 1),
('iPhone 14 Pro', 'iOS 16', 2022, 999, 3200, 6.1, 206, 2),
('Pixel 7', 'Android 13', 2022, 599, 4614, 6.4, 197, 3),
('Galaxy A52', 'Android 11', 2021, 499, 4500, 6.5, 189, 1),
('iPhone 12', 'iOS 14', 2020, 699, 2815, 6.1, 164, 2),
('Pixel 5', 'Android 11', 2020, 699, 4080, 6.0, 151, 3),
('Galaxy Fold 3', 'Android 12', 2021, 1799, 4400, 7.6, 271, 1),
('iPhone XR', 'iOS 13', 2019, 499, 2942, 6.1, 194, 2),
('Pixel 4a', 'Android 11', 2020, 349, 3140, 5.8, 143, 3),
('Galaxy S23', 'Android 14', 2023, 799, 4700, 6.1, 168, 1),
('iPhone 15', 'iOS 17', 2023, 1099, 3279, 6.1, 206, 2),
('Pixel 8', 'Android 14', 2023, 699, 4575, 6.2, 193, 3),
('Galaxy Z Flip 4', 'Android 13', 2022, 999, 3700, 6.7, 187, 1),
('iPhone 13 Mini', 'iOS 15', 2021, 699, 2438, 5.4, 140, 2),
('Pixel 6a', 'Android 13', 2022, 449, 4410, 6.1, 178, 3),
('Galaxy M31', 'Android 10', 2020, 279, 6000, 6.4, 191, 1),
('iPhone 11', 'iOS 13', 2019, 699, 3110, 6.1, 194, 2),
('Pixel 3 XL', 'Android 10', 2019, 899, 3430, 6.3, 184, 3),
('Galaxy Note 10', 'Android 10', 2019, 949, 3500, 6.3, 168, 1),
('iPhone XS Max', 'iOS 12', 2018, 1099, 3174, 6.5, 208, 2);
