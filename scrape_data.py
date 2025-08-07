from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import csv
import time


def setup_driver():
    """Set up Chrome driver with basic options"""
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)
    return driver


def scrape_part_info(driver, part_name, product_url):
    """
    Scrape detailed information from a product page
    """
    # Initialize data structure
    data = {
        "part_name": part_name,
        "part_id": "N/A",
        "mpn_id": "N/A",
        "part_price": "N/A",
        "install_difficulty": "N/A",
        "install_time": "N/A",
        "symptoms": "N/A",
        "product_types": "N/A",
        "brand": "N/A",
        "availability": "N/A",
        "product_url": product_url,
    }

    try:
        # Navigate to product page
        driver.get(product_url)
        wait = WebDriverWait(driver, 15)

        # Wait for page to load
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.pd__wrap")))

        # Extract Product ID
        try:
            product_id = driver.find_element(
                By.CSS_SELECTOR, "span[itemprop='productID']"
            )
            data["part_id"] = product_id.text
        except:
            pass

        # Extract MPN
        try:
            mpn = driver.find_element(By.CSS_SELECTOR, "span[itemprop='mpn']")
            data["mpn_id"] = mpn.text
        except:
            pass

        # Extract Brand
        try:
            brand = driver.find_element(
                By.CSS_SELECTOR, "span[itemprop='brand'] span[itemprop='name']"
            )
            data["brand"] = brand.text
        except:
            pass

        # Extract Availability
        try:
            availability = driver.find_element(
                By.CSS_SELECTOR, "span[itemprop='availability']"
            )
            data["availability"] = availability.text
        except:
            pass

        # Extract Price (multiple attempts)
        try:
            price_container = driver.find_element(
                By.CSS_SELECTOR, "span.price.pd__price"
            )
            # Try direct price element first
            try:
                price_element = price_container.find_element(
                    By.CSS_SELECTOR, "span.js-partPrice"
                )
                data["part_price"] = price_element.text
            except:
                # Fallback to container text
                data["part_price"] = price_container.text
        except:
            pass

        # Extract Installation Info
        try:
            install_container = driver.find_element(
                By.CSS_SELECTOR, "div.d-flex.flex-lg-grow-1"
            )
            d_flex_divs = install_container.find_elements(By.CLASS_NAME, "d-flex")

            if len(d_flex_divs) >= 2:
                # Difficulty
                try:
                    difficulty_p = d_flex_divs[0].find_element(By.TAG_NAME, "p")
                    data["install_difficulty"] = difficulty_p.text
                except:
                    pass

                # Time
                try:
                    time_p = d_flex_divs[1].find_element(By.TAG_NAME, "p")
                    data["install_time"] = time_p.text
                except:
                    pass
        except:
            pass

        # Extract Symptoms and Product Types
        try:
            pd_wrap = driver.find_element(By.CSS_SELECTOR, "div.pd__wrap.row")
            info_divs = pd_wrap.find_elements(By.CSS_SELECTOR, "div.col-md-6.mt-3")

            for div in info_divs:
                try:
                    header = div.find_element(By.CSS_SELECTOR, "div.bold.mb-1")
                    header_text = header.text
                    full_text = div.text

                    if "This part fixes the following symptoms:" in header_text:
                        data["symptoms"] = full_text.replace(header_text, "").strip()
                    elif "This part works with the following products:" in header_text:
                        data["product_types"] = full_text.replace(
                            header_text, ""
                        ).strip()
                except:
                    continue
        except:
            pass

    except Exception as e:
        print(f"Error scraping {part_name}: {e}")

    return data


def process_category_page(driver, category_url):
    """
    Process a category page and extract all product links
    """
    print(f"Processing category: {category_url}")
    parts_data = []

    try:
        # Navigate to category page
        driver.get(category_url)
        wait = WebDriverWait(driver, 15)

        # Wait for product listings to load
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.nf__part.mb-3"))
        )

        # Find all product containers
        part_divs = driver.find_elements(By.CSS_SELECTOR, "div.nf__part.mb-3")
        print(f"Found {len(part_divs)} products in category")

        # Extract product info to avoid stale elements
        product_links = []
        for part_div in part_divs:
            try:
                title_link = part_div.find_element(
                    By.CLASS_NAME, "nf__part__detail__title"
                )
                part_name_span = title_link.find_element(By.TAG_NAME, "span")
                part_name = part_name_span.text
                product_url = title_link.get_attribute("href")

                if product_url:
                    product_links.append((part_name, product_url))
            except:
                continue

        # Process each product
        parts_data = process_parts_in_category(driver, product_links, category_url)

    except Exception as e:
        print(f"Error processing category {category_url}: {e}")

    return parts_data


def process_parts_in_category(driver, product_links, category_url):
    """
    Process individual parts within a category
    """
    parts_data = []

    for i, (part_name, product_url) in enumerate(product_links, 1):
        print(f"  Processing part {i}/{len(product_links)}: {part_name}")

        # Scrape individual product
        part_data = scrape_part_info(driver, part_name, product_url)
        parts_data.append(part_data)

        # Navigate back to category page for next product
        try:
            driver.get(category_url)
            time.sleep(1)  # Brief pause
        except:
            print(f"Failed to return to category page. Stopping category processing.")
            break

    return parts_data


def get_related_links(driver, main_page_url):
    """
    Get related category links from the main appliance page
    """
    related_links = []

    try:
        # Navigate to the main page (e.g., Dishwasher-Parts.htm)
        driver.get(main_page_url)
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "section-title")))

        section_titles = driver.find_elements(By.CLASS_NAME, "section-title")

        for title in section_titles:
            title_text = title.text
            # Look for "Related" sections for the specific appliance type
            if "Related" in title_text and (
                "Dishwasher Parts" in title_text or "Refrigerator Parts" in title_text
            ):
                print(f"Found related section: {title_text}")
                try:
                    # Find next ul.nf__links after this title
                    related_ul = title.find_element(
                        By.XPATH, "./following::ul[@class='nf__links'][1]"
                    )
                    li_tags = related_ul.find_elements(By.TAG_NAME, "li")

                    for li_tag in li_tags:
                        try:
                            a_tag = li_tag.find_element(By.TAG_NAME, "a")
                            link_url = a_tag.get_attribute("href")
                            link_text = a_tag.text
                            if link_url:
                                related_links.append(link_url)
                                print(f"  Found category: {link_text} -> {link_url}")
                        except:
                            continue
                except:
                    continue

    except Exception as e:
        print(f"Error getting related links: {e}")

    return related_links


def scrape_related_categories(main_page_url):
    """
    Main function to scrape all parts from related categories only
    """
    print(f"Starting scraper for related categories from: {main_page_url}")
    all_parts_data = []
    driver = None

    try:
        # Step 1: Get all related category links
        print("Getting related category links...")
        driver = setup_driver()
        related_links = get_related_links(driver, main_page_url)

        if not related_links:
            print("No related categories found!")
            return all_parts_data

        print(f"Found {len(related_links)} related categories to process")

        # Step 2: Process each related category
        for i, category_url in enumerate(related_links, 1):
            print(f"\n{'='*60}")
            print(f"PROCESSING CATEGORY {i}/{len(related_links)}")
            print(f"URL: {category_url}")
            print(f"{'='*60}")

            category_data = process_category_page(driver, category_url)
            all_parts_data.extend(category_data)

            print(f"\nCategory complete: {len(category_data)} products found")
            print(f"Progress: {i}/{len(related_links)} categories completed")
            print(f"Total products found so far: {len(all_parts_data)}")

            # Small delay between categories
            time.sleep(1)

    except Exception as e:
        print(f"Error in main scraping process: {e}")
    finally:
        if driver:
            driver.quit()

    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE!")
    print(f"Total products found: {len(all_parts_data)}")
    print(f"{'='*60}")

    return all_parts_data


def save_to_csv(parts_data, filename):
    """Save parts data to CSV file"""
    if not parts_data:
        print("No data to save.")
        return

    try:
        fieldnames = parts_data[0].keys()
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(parts_data)

        print(f"Successfully saved {len(parts_data)} parts to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")


# Example usage
if __name__ == "__main__":
    main_page_url = "https://www.partselect.com/Dishwasher-Parts.htm"
    parts_data = scrape_related_categories(main_page_url)
    save_to_csv(parts_data, "dishwasher_parts.csv")

    main_page_url = "https://www.partselect.com/Refrigerator-Parts.htm"
    parts_data = scrape_related_categories(main_page_url)
    save_to_csv(parts_data, "refrigerator_parts.csv")
