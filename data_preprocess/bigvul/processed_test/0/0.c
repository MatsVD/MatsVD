static bool check_rodc_critical_attribute(struct ldb_message *msg)
{
	uint32_t schemaFlagsEx, searchFlags, rodc_filtered_flags;
	schemaFlagsEx = ldb_msg_find_attr_as_uint(msg, "schemaFlagsEx", 0);
	searchFlags = ldb_msg_find_attr_as_uint(msg, "searchFlags", 0);
	rodc_filtered_flags = (SEARCH_FLAG_RODC_ATTRIBUTE
			      | SEARCH_FLAG_CONFIDENTIAL);
	if ((schemaFlagsEx & SCHEMA_FLAG_ATTR_IS_CRITICAL) &&
		((searchFlags & rodc_filtered_flags) == rodc_filtered_flags)) {
		return true;
	} else {
		return false;
	}
}