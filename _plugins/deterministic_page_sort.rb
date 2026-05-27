require "date"
require "time"

module DeterministicPageSort
  LEAF_ORDER = {
    "study:diffusion-tutorial" => 1,
    "study:algorithm" => 2,
    "study:system-optimization" => 3,
    "study:inference-systems" => 4,
    "paper:inference" => 5,
    "paper:success-rate" => 6,
    "project:shallow-pi" => 7,
    "work-log:" => 8,
  }.freeze

  def sort_pages_by_date(input)
    sort_pages(input, include_leaf_order: false)
  end

  def sort_pages_for_home(input)
    sort_pages(input, include_leaf_order: true)
  end

  private

  def sort_pages(input, include_leaf_order:)
    Array(input).sort_by do |item|
      key = [-timestamp_for(page_value(item, "date"))]
      key << LEAF_ORDER.fetch(leaf_key(item), 999) if include_leaf_order
      key + [sort_text(page_value(item, "title")), page_value(item, "url").to_s]
    end
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
end

Liquid::Template.register_filter(DeterministicPageSort)
