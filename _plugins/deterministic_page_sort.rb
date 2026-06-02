require "date"
require "time"

module DeterministicPageSort
  def sort_pages_by_date(input)
    sort_pages(input, include_leaf_order: false)
  end

  def sort_pages_for_home(input)
    sort_pages(input, include_leaf_order: true)
  end

  private

  def sort_pages(input, include_leaf_order:)
    leaf_order = include_leaf_order ? sidebar_leaf_order : {}

    Array(input).sort_by do |item|
      key = [-timestamp_for(page_value(item, "date"))]
      key << leaf_order.fetch(leaf_key(item), 999) if include_leaf_order
      key + [sort_text(page_value(item, "title")), page_value(item, "url").to_s]
    end
  end

  def sidebar_leaf_order
    site = liquid_site
    site ? build_sidebar_leaf_order(site) : {}
  end

  def build_sidebar_leaf_order(site)
    content_keys = site.pages.each_with_object({}) do |page, keys|
      next unless page_value(page, "date")

      keys[leaf_key(page)] = true
    end

    nav_pages = site.pages.select { |page| nav_page?(page) }
    pages_by_title = nav_pages.each_with_object({}) do |page, pages|
      pages[page_value(page, "title")] ||= page
    end

    nav_pages
      .select { |page| content_keys[leaf_key_for_nav_page(page)] }
      .sort_by { |page| sidebar_sort_key(page, pages_by_title) }
      .each_with_index
      .to_h { |page, index| [leaf_key_for_nav_page(page), index] }
  end

  def liquid_site
    @context.registers[:site] if defined?(@context) && @context
  end

  def nav_page?(page)
    page_value(page, "title") &&
      page_value(page, "date").nil? &&
      page_value(page, "nav_exclude") != true
  end

  def leaf_key_for_nav_page(page)
    segments = page_value(page, "url").to_s.split("/").reject(&:empty?)
    return "" if segments.empty?

    section = segments.first
    subcategory = segments.length == 1 ? "" : segments.last
    "#{section}:#{subcategory}"
  end

  def sidebar_sort_key(page, pages_by_title)
    ancestor_chain(page, pages_by_title).flat_map do |nav_page|
      [
        nav_order(page_value(nav_page, "nav_order")),
        sort_text(page_value(nav_page, "title")),
        page_value(nav_page, "url").to_s,
      ]
    end
  end

  def ancestor_chain(page, pages_by_title)
    [
      pages_by_title[page_value(page, "grand_parent")],
      pages_by_title[page_value(page, "parent")],
      page,
    ].compact.uniq
  end

  def leaf_key(item)
    "#{page_value(item, "section")}:#{page_value(item, "subcategory")}"
  end

  def page_value(item, key)
    return item_property(item, key) if respond_to?(:item_property, true)

    value = hash_value(item, key)
    return value unless value.nil?

    if item.respond_to?(:data)
      value = hash_value(item.data, key)
      return value unless value.nil?
    end

    item.public_send(key) if item.respond_to?(key)
  end

  def hash_value(item, key)
    return unless item.respond_to?(:[])

    item[key]
  rescue StandardError
    nil
  end

  def timestamp_for(value)
    case value
    when Time
      value.to_i
    when DateTime
      value.to_time.to_i
    when Date
      Time.new(value.year, value.month, value.day).to_i
    else
      Time.parse(value.to_s).to_i
    end
  rescue ArgumentError, TypeError
    0
  end

  def sort_text(value)
    value.to_s.downcase
  end

  def nav_order(value)
    value.nil? ? 999 : value.to_f
  rescue ArgumentError, TypeError
    999
  end
end

Liquid::Template.register_filter(DeterministicPageSort)
