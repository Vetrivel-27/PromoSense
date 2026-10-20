-- PromoSense Schema Migration 001: Initial Relational & Analytical Schema

-- 1. Households
CREATE TABLE IF NOT EXISTS households (
    household_key BIGINT PRIMARY KEY,
    age_band VARCHAR(50),
    marital_status VARCHAR(20),
    income_band VARCHAR(50),
    homeowner_status VARCHAR(50),
    household_comp VARCHAR(100),
    household_size VARCHAR(20),
    kid_category VARCHAR(50),
    has_demographics BOOLEAN NOT NULL DEFAULT FALSE
);

-- 2. Products
CREATE TABLE IF NOT EXISTS products (
    product_id BIGINT PRIMARY KEY,
    manufacturer_id BIGINT NOT NULL,
    department VARCHAR(100) NOT NULL,
    brand_type VARCHAR(50) NOT NULL,
    commodity VARCHAR(100) NOT NULL,
    sub_commodity VARCHAR(100) NOT NULL,
    size_description VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_products_dept ON products(department);
CREATE INDEX IF NOT EXISTS idx_products_commodity ON products(commodity);
CREATE INDEX IF NOT EXISTS idx_products_manufacturer ON products(manufacturer_id);

-- 3. Stores
CREATE TABLE IF NOT EXISTS stores (
    store_id INT PRIMARY KEY
);

-- 4. Campaigns
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id INT PRIMARY KEY,
    campaign_type VARCHAR(50) NOT NULL,
    start_day INT NOT NULL,
    end_day INT NOT NULL,
    start_week INT NOT NULL,
    end_week INT NOT NULL,
    duration_days INT NOT NULL
);

-- 5. Campaign Assignments (Treatment Assignment)
CREATE TABLE IF NOT EXISTS campaign_assignments (
    id BIGSERIAL PRIMARY KEY,
    household_key BIGINT NOT NULL REFERENCES households(household_key),
    campaign_id INT NOT NULL REFERENCES campaigns(campaign_id),
    campaign_type VARCHAR(50) NOT NULL,
    CONSTRAINT uq_assignment_hh_camp UNIQUE (household_key, campaign_id)
);

CREATE INDEX IF NOT EXISTS idx_assignment_hh ON campaign_assignments(household_key);
CREATE INDEX IF NOT EXISTS idx_assignment_camp ON campaign_assignments(campaign_id);

-- 6. Coupons
CREATE TABLE IF NOT EXISTS coupons (
    id BIGSERIAL PRIMARY KEY,
    coupon_upc VARCHAR(50) NOT NULL,
    product_id BIGINT NOT NULL REFERENCES products(product_id),
    campaign_id INT NOT NULL REFERENCES campaigns(campaign_id)
);

CREATE INDEX IF NOT EXISTS idx_coupon_upc ON coupons(coupon_upc);
CREATE INDEX IF NOT EXISTS idx_coupon_product ON coupons(product_id);
CREATE INDEX IF NOT EXISTS idx_coupon_camp ON coupons(campaign_id);

-- 7. Coupon Redemptions
CREATE TABLE IF NOT EXISTS coupon_redemptions (
    id BIGSERIAL PRIMARY KEY,
    household_key BIGINT NOT NULL REFERENCES households(household_key),
    day INT NOT NULL,
    week_no INT NOT NULL,
    coupon_upc VARCHAR(50) NOT NULL,
    campaign_id INT NOT NULL REFERENCES campaigns(campaign_id)
);

CREATE INDEX IF NOT EXISTS idx_redempt_hh ON coupon_redemptions(household_key);
CREATE INDEX IF NOT EXISTS idx_redempt_day ON coupon_redemptions(day);
CREATE INDEX IF NOT EXISTS idx_redempt_camp ON coupon_redemptions(campaign_id);

-- 8. Transactions
CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    household_key BIGINT NOT NULL REFERENCES households(household_key),
    basket_id VARCHAR(50) NOT NULL,
    day INT NOT NULL,
    week_no INT NOT NULL,
    product_id BIGINT NOT NULL REFERENCES products(product_id),
    store_id INT NOT NULL REFERENCES stores(store_id),
    quantity DOUBLE PRECISION NOT NULL,
    sales_value DOUBLE PRECISION NOT NULL,
    retail_disc DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    coupon_disc DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    coupon_match_disc DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    trans_time INT,
    shelf_price DOUBLE PRECISION,
    paid_unit_price DOUBLE PRECISION,
    retail_discount_pct DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    in_store_promoted BOOLEAN NOT NULL DEFAULT FALSE,
    coupon_used BOOLEAN NOT NULL DEFAULT FALSE,
    row_classification VARCHAR(50) NOT NULL DEFAULT 'valid_purchase'
);

CREATE INDEX IF NOT EXISTS idx_txn_hh_week ON transactions(household_key, week_no);
CREATE INDEX IF NOT EXISTS idx_txn_prod_week ON transactions(product_id, week_no);
CREATE INDEX IF NOT EXISTS idx_txn_store_prod_week ON transactions(store_id, product_id, week_no);
CREATE INDEX IF NOT EXISTS idx_txn_basket ON transactions(basket_id);

-- 9. Store Promotions (Causal display & mailer features)
CREATE TABLE IF NOT EXISTS store_promotions (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL REFERENCES products(product_id),
    store_id INT NOT NULL REFERENCES stores(store_id),
    week_no INT NOT NULL,
    display_code VARCHAR(20) NOT NULL,
    mailer_code VARCHAR(20) NOT NULL,
    has_display_placement BOOLEAN NOT NULL DEFAULT FALSE,
    has_mailer_placement BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_promo_prod_store_week ON store_promotions(product_id, store_id, week_no);
