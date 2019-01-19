package oneshot

import (
	"sync"
)

var oneshots = &sync.Map{}

func Do(name string, fn func()) {
	o, _ := oneshots.LoadOrStore(name, &sync.Once{})
	if once, ok := o.(*sync.Once); ok {
		once.Do(log(name, fn))
	}
}

func Reset(names ...string) {
	if len(names) == 0 {
		oneshots = &sync.Map{}
		return
	}

	for _, n := range names {
		oneshots.Delete(n)
		oneshots.Delete(deprecated + n)
	}
}
