#                func_logit = []
#                func_lgt_list = []
#
#                for name in s.possible_names:
#                    for tk_id in name:
#                        func_logit.append(s.logits[tk_id])
#                    func_lgt_list.append(func_logit.copy())
#                    func_logit.clear()
#                sum_lgt_list = [sum(func_lgt) for func_lgt in func_lgt_list]
#                possible_name = max(sum_lgt_list)
#                idx_possible_name = sum_lgt_list.index(possible_name)
#                input_ids.extend(s.possible_names[idx_possible_name])
#                input_ids.extend([aspas, comma, space, aspas, parameters, aspas, colon, space, open_chav, aspas])
#                count += len(s.possible_names[idx_possible_name]) + 10
#                continue
