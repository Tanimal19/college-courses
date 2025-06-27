#ifndef UTIL_HEADER
#define UTIL_HEADER

#include <time.h>

#include "def.h"

/* linked list for segments */
enum class SegmentState {
    UNSENT,
    SENT_UNACKED,
    ACKED
};

struct segment_node {
    struct segment sgmt;
    struct segment_node *next;
    SegmentState state;
};

class SegmentQueue {
    private:
        struct segment_node *head;
        struct segment_node *tail;
        int size;

    public:
        SegmentQueue() : head(nullptr), tail(nullptr), size(0) {}

        ~SegmentQueue() {
            while (head != nullptr) {
                struct segment_node *temp = head;
                head = head->next;
                delete temp;
            }
        }

        struct segment_node* get_by_index(int index) {
            if (index < 0 || index >= size) {
                return nullptr;
            }

            struct segment_node *cur = head;
            for (int i = 0; i < index; ++i) {
                cur = cur->next;
            }
            return cur;
        }

        struct segment_node* get_by_seqNumber(int seqNumber) {
            struct segment_node *cur = head;
            while (cur != nullptr) {
                if (cur->sgmt.head.seqNumber == seqNumber) {
                    return cur;
                }
                cur = cur->next;
            }
            return nullptr;
        }

        void add(struct segment sgmt) {
            struct segment_node *new_node = new segment_node{sgmt, nullptr, SegmentState::UNSENT};
            if (tail == nullptr) {
                head = tail = new_node;
            } else {
                tail->next = new_node;
                tail = new_node;
            }
            size++;
        }

        bool remove(int seqNumber) {
            struct segment_node *cur = head;
            struct segment_node *prev = nullptr;

            while (cur != nullptr) {
                if (cur->sgmt.head.seqNumber == seqNumber) {
                    // only remove segments that have been ACKed
                    if (cur->state != SegmentState::ACKED) {
                        return false;
                    }

                    if (prev == nullptr) {
                        head = cur->next;
                    } else {
                        prev->next = cur->next;
                    }
                    if (cur == tail) {
                        tail = prev;
                    }
                    delete cur;
                    size--;
                    return true;
                }
                prev = cur;
                cur = cur->next;
            }
            return false;
        }

        int get_size() const {
            return size;
        }

        bool is_empty() const {
            return size == 0;
        }
};

#endif // UTIL_HEADER