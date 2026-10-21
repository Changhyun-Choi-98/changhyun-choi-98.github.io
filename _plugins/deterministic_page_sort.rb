require "date"
require "time"

module DeterministicPageSort
  def sort_pages_by_date(input)
    sort_pages(input, include_leaf_order: false)
  end

  def sort_pages_for_home(input)
    sort_pages(input, include_leaf_order: true)
  end

  def post_category_path(input)
    category_paths.fetch(leaf_key(input), []).map { |page| page_value(page, "title") }.join(" / ")
  end

  def nav_post_count(input)
    site = liquid_site
    return 0 unless site

    posts = site.pages.select { |page| page_value(page, "date") }
    return posts.size if page_value(input, "count_all_posts") == true

    paths = category_paths
    posts.count do |post|
      paths.fetch(leaf_key(post), []).any? do |ancestor|
        page_value(ancestor, "url") == page_value(input, "url")
      end
    end
  end

  private

  def category_paths
    site = liquid_site
    return {} unless site

    nav_pages = site.pages.select { |page| nav_page?(page) }
    pages_by_title = nav_pages.group_by { |page| page_value(page, "title") }
    nav_pages.to_h { |page| [leaf_key_for_nav_page(page), ancestor_chain(page, pages_by_title)] }
  end

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
    pages_by_title = nav_pages.group_by { |page| page_value(page, "title") }

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
    chain = []
    while page && !chain.include?(page)
      chain.unshift(page)
      parents = pages_by_title.fetch(page_value(page, "parent"), [])
      page = parents.find { |parent| page_value(parent, "parent") == page_value(page, "grand_parent") } || parents.first
    end
    chain
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
